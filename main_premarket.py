#!/usr/bin/env python3
"""
미국 주식 시장 프리마켓 리포트 생성 및 발송

장 시작 전(KST 21:00, EST 7:00 AM) 프리마켓 가격 + 최신 뉴스 + 전일 기술적 분석 기반 리포트

Usage:
    python main_premarket.py              # 전체 실행 (리포트 생성 + Slack 발송)
    python main_premarket.py --dry-run    # 발송 없이 리포트만 생성
"""

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from config.sp500_tickers import MARKET_ETFS, SECTOR_ETFS, get_all_tickers
from data.fetcher import StockDataFetcher
from data.premarket import PreMarketFetcher
from data.premarket_tickers import (
    load_previous_recommendations,
    get_premarket_tickers,
)
from data.calendar import EconomicCalendar, EarningsCalendar
from analysis.technical import TechnicalAnalyzer
from analysis.signals import SignalDetector
from analysis.sentiment import NewsSentimentAnalyzer
from news.fetcher import NewsFetcher
from report.premarket_generator import PreMarketReportGenerator
from notification.slack_sender import SlackSender


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, settings.general.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(dry_run: bool = False):
    """프리마켓 리포트 메인 실행 함수"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("미국 주식 프리마켓 리포트 생성 시작")
    logger.info("=" * 50)

    try:
        # 1. 대상 종목 결정
        logger.info("1. 대상 종목 결정 중...")
        all_tickers = get_premarket_tickers()
        prev_rec_tickers = load_previous_recommendations()
        logger.info(f"   대상 종목 합계: {len(all_tickers)}개")

        # 2. 전일 OHLCV 배치 수집 (기술적 분석용)
        logger.info("2. 전일 OHLCV 데이터 수집 중...")
        fetcher = StockDataFetcher()
        stock_data = fetcher.fetch_batch(all_tickers)
        logger.info(f"   {len(stock_data)}개 종목 데이터 수집 완료")

        # 시장 요약 (전일 종가 기준)
        market_summary = fetcher.get_market_summary()

        # 3. 프리마켓 가격 수집
        logger.info("3. 프리마켓 가격 수집 중...")
        pm_fetcher = PreMarketFetcher()
        premarket_data = pm_fetcher.fetch_batch(all_tickers)

        # 시장 ETF 프리마켓
        market_premarket = {t: asdict(premarket_data[t]) for t in MARKET_ETFS if t in premarket_data}

        # 섹터 ETF 프리마켓
        sector_etf_list = list(SECTOR_ETFS.values())
        sector_premarket = {t: asdict(premarket_data[t]) for t in sector_etf_list if t in premarket_data}

        # 주요 변동 종목
        significant_movers_raw = pm_fetcher.get_significant_movers(premarket_data)
        significant_movers = {
            "gainers": [asdict(g) for g in significant_movers_raw["gainers"]],
            "losers": [asdict(l) for l in significant_movers_raw["losers"]],
        }
        logger.info(f"   주요 변동: 상승 {len(significant_movers['gainers'])}개, 하락 {len(significant_movers['losers'])}개")

        # 4. 전일 기술적 분석 (추천 종목 + 변동 종목)
        logger.info("4. 전일 기술적 분석 수행 중...")
        tech_analyzer = TechnicalAnalyzer()
        spy_df = stock_data.get("SPY")
        analysis_results = tech_analyzer.analyze_batch(stock_data, spy_df)
        logger.info(f"   {len(analysis_results)}개 종목 분석 완료")

        # 5. 전일 추천 종목 현황 구성
        logger.info("5. 전일 추천 종목 현황 구성 중...")
        previous_recommendations = []
        for ticker in prev_rec_tickers:
            pm = premarket_data.get(ticker)
            analysis = analysis_results.get(ticker, {})

            rec_info = {
                "ticker": ticker,
                "method": None,
                "has_premarket": pm.has_premarket if pm else False,
                "pm_price": pm.pre_market_price if pm and pm.has_premarket else None,
                "pm_change_pct": pm.pre_market_change_pct if pm and pm.has_premarket else None,
                "prev_close": pm.regular_previous_close if pm else None,
                "analysis": None,
                "sentiment": None,
            }

            # 기술적 분석 오버레이
            if analysis:
                rsi = analysis.get("rsi")
                macd = analysis.get("macd")
                kalman = analysis.get("kalman")
                adx = analysis.get("adx")
                bollinger = analysis.get("bollinger")
                volume = analysis.get("volume")

                rec_info["analysis"] = {
                    "rsi": round(rsi.value, 1) if rsi else None,
                    "macd_signal": "골든크로스" if macd and macd.is_bullish_cross else ("데드크로스" if macd and not macd.is_bullish_cross else None),
                    "kalman_price": round(kalman.predicted_price, 2) if kalman else None,
                    "adx": round(adx.adx, 1) if adx else None,
                    "bollinger_z": round(bollinger.z_score, 2) if bollinger else None,
                    "volume_ratio": round(volume.volume_ratio, 2) if volume else None,
                }

            previous_recommendations.append(rec_info)
        logger.info(f"   전일 추천 종목 {len(previous_recommendations)}개 구성")

        # 6. 뉴스 수집
        logger.info("6. 뉴스 수집 중...")
        news_fetcher = NewsFetcher()
        news = {"market_news": news_fetcher.fetch_market_news()}
        logger.info(f"   {len(news.get('market_news', []))}개 시장 뉴스 수집")

        # 6-1. 주요 변동 종목 감성 분석
        ticker_news_map = {}
        sentiment_results = {}
        try:
            logger.info("6-1. 주요 변동 종목 감성 분석 중...")
            sentiment_tickers = []
            for g in significant_movers_raw["gainers"][:3]:
                sentiment_tickers.append(g.ticker)
            for l in significant_movers_raw["losers"][:3]:
                sentiment_tickers.append(l.ticker)
            # 전일 추천 종목도 포함
            for t in prev_rec_tickers:
                if t not in sentiment_tickers:
                    sentiment_tickers.append(t)

            if sentiment_tickers:
                ticker_news_map = {}
                for t in sentiment_tickers:
                    t_news = news_fetcher.fetch_ticker_news(t)
                    if t_news:
                        ticker_news_map[t] = t_news

                sentiment_analyzer = NewsSentimentAnalyzer()
                sentiment_results = sentiment_analyzer.analyze_multiple_tickers(ticker_news_map)
                logger.info(f"   {len(sentiment_results)}개 종목 감성 분석 완료")

                # 추천 종목에 감성 주입
                for rec in previous_recommendations:
                    ticker = rec["ticker"]
                    if ticker in sentiment_results:
                        ts = sentiment_results[ticker]
                        rec["sentiment"] = {
                            "label": ts.label,
                            "gauge_score": ts.gauge_score,
                            "positive_count": ts.positive_count,
                            "negative_count": ts.negative_count,
                        }

                # 변동 종목에 감성 배지 정보 추가
                for key in ("gainers", "losers"):
                    for mover in significant_movers[key]:
                        t = mover["ticker"]
                        if t in sentiment_results:
                            ts = sentiment_results[t]
                            mover["sentiment_label"] = ts.label
            else:
                logger.info("   감성 분석 대상 종목 없음")
        except Exception as e:
            logger.warning(f"   감성 분석 실패 (리포트 생성 계속): {e}")

        # 6-2. 개장 급등 추천 (S&P 500 PM gainer만 분석)
        opening_surge_recommendations = []
        try:
            logger.info("6-2. 개장 급등 추천 — S&P 500 PM gainer 스캔 중...")
            min_pm_pct = settings.analysis.surge_min_pm_change_pct

            # 6-2a. yf.download 배치로 PM 상승 종목 빠르게 스캔 (2회 HTTP 호출)
            sp500_tickers = get_all_tickers()
            surge_gainer_tickers = pm_fetcher.scan_premarket_gainers(
                sp500_tickers, min_change_pct=min_pm_pct,
            )

            if surge_gainer_tickers:
                # 6-2b. gainer 종목만 상세 PM 데이터 수집 (개별 info 호출)
                new_pm_tickers = [t for t in surge_gainer_tickers if t not in premarket_data]
                if new_pm_tickers:
                    logger.info(f"   PM gainer {len(new_pm_tickers)}개 상세 데이터 수집 중...")
                    new_pm = pm_fetcher.fetch_batch(new_pm_tickers)
                    premarket_data.update(new_pm)

                # 6-2c. gainer 종목 OHLCV + 기술적 분석 (미분석 종목만)
                analyze_tickers = [t for t in surge_gainer_tickers if t not in analysis_results]
                if analyze_tickers:
                    logger.info(f"   {len(analyze_tickers)}개 종목 기술적 분석 중...")
                    surge_stock_data = fetcher.fetch_batch(analyze_tickers)
                    surge_analysis = TechnicalAnalyzer().analyze_batch(surge_stock_data, spy_df)
                    analysis_results.update(surge_analysis)

                # 6-2d. gainer 종목 뉴스 감성 분석 (미분석 종목만)
                surge_sentiment = {}
                surge_news_tickers = [t for t in surge_gainer_tickers if t not in sentiment_results]
                if surge_news_tickers:
                    surge_news_map = {}
                    for t in surge_news_tickers:
                        if t in ticker_news_map:
                            surge_news_map[t] = ticker_news_map[t]
                        else:
                            t_news = news_fetcher.fetch_ticker_news(t)
                            if t_news:
                                surge_news_map[t] = t_news
                    if surge_news_map:
                        surge_analyzer = NewsSentimentAnalyzer()
                        surge_sentiment = surge_analyzer.analyze_multiple_tickers(surge_news_map)

                all_surge_sentiment = dict(sentiment_results)
                all_surge_sentiment.update(surge_sentiment)
            else:
                logger.info("   PM 상승 종목 없음")
                all_surge_sentiment = dict(sentiment_results)

            # 6-2e. 개장 급등 추천 실행
            signal_detector = SignalDetector(analysis_results)
            opening_surge_recommendations = signal_detector.get_opening_surge_recommendations(
                premarket_data=premarket_data,
                sentiment_results=all_surge_sentiment,
            )

            # 추천 종목에 sentiment dict 주입
            for rec in opening_surge_recommendations:
                t = rec["ticker"]
                if rec.get("sentiment") is None and t in all_surge_sentiment:
                    ts = all_surge_sentiment[t]
                    rec["sentiment"] = {
                        "label": ts.label,
                        "gauge_score": ts.gauge_score,
                        "avg_compound": ts.avg_compound,
                        "positive_count": ts.positive_count,
                        "negative_count": ts.negative_count,
                    }

            logger.info(f"   개장 급등 추천 {len(opening_surge_recommendations)}개 종목 선정")
        except Exception as e:
            logger.warning(f"   개장 급등 추천 실패 (리포트 생성 계속): {e}")
            opening_surge_recommendations = []

        # 7. 경제/실적 캘린더
        logger.info("7. 경제/실적 캘린더 수집 중...")
        economic_cal = EconomicCalendar()
        economic_calendar = economic_cal.get_week_calendar()

        earnings_cal = EarningsCalendar()
        earnings_calendar = earnings_cal.get_week_calendar()
        logger.info("   캘린더 수집 완료")

        # 8. 프리마켓 리포트 생성
        logger.info("8. 프리마켓 리포트 생성 중...")
        report_gen = PreMarketReportGenerator()
        html_report = report_gen.generate(
            market_premarket=market_premarket,
            sector_premarket=sector_premarket,
            significant_movers=significant_movers,
            previous_recommendations=previous_recommendations,
            economic_calendar=economic_calendar,
            earnings_calendar=earnings_calendar,
            news=news,
            opening_surge_recommendations=opening_surge_recommendations,
        )

        report_path = report_gen.save_to_file(html_report)
        logger.info(f"   리포트 저장: {report_path}")

        # 8-1. 개장 급등 추천 기록
        try:
            from tracking import RecommendationRecorder
            recorder = RecommendationRecorder()
            surge_count = recorder.record_surge(opening_surge_recommendations)
            logger.info(f"   개장 급등 추천 {surge_count}건 기록 완료")
        except Exception as e:
            logger.warning(f"   개장 급등 추천 기록 실패: {e}")

        # 9. Slack 발송
        if dry_run:
            logger.info("9. [DRY-RUN] Slack 발송 건너뜀")
            logger.info(f"   생성된 리포트를 확인하세요: {report_path}")
        else:
            logger.info("9. Slack 발송 중...")
            slack_sender = SlackSender()
            success = slack_sender.send_premarket(
                html_content=html_report,
                market_summary=market_summary,
                premarket_data=market_premarket,
                significant_movers=significant_movers,
                opening_surge_recommendations=opening_surge_recommendations,
            )

            if success:
                logger.info("   Slack 발송 완료!")
            else:
                logger.error("   Slack 발송 실패")
                sys.exit(1)

        logger.info("=" * 50)
        logger.info("프리마켓 리포트 생성 완료!")
        logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="미국 주식 프리마켓 리포트 생성"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="발송 없이 리포트만 생성",
    )

    args = parser.parse_args()
    main(dry_run=args.dry_run)
