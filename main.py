#!/usr/bin/env python3
"""
미국 주식 시장 일일 리포트 생성 및 발송

Usage:
    python main.py              # 전체 실행 (리포트 생성 + Slack 발송)
    python main.py --dry-run    # 발송 없이 리포트만 생성
    python main.py --test-slack # 테스트 Slack 메시지 발송
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from config.sp500_tickers import get_all_tickers, get_ticker_info, get_sp500_by_sector
from data.fetcher import StockDataFetcher
from data.calendar import EconomicCalendar, EarningsCalendar
from analysis.technical import TechnicalAnalyzer
from analysis.sector import SectorAnalyzer
from analysis.signals import SignalDetector
from analysis.sentiment import NewsSentimentAnalyzer
from news.fetcher import NewsFetcher
from report.generator import ReportGenerator
# from notification.email_sender import EmailSender
from notification.slack_sender import SlackSender


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, settings.general.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(dry_run: bool = False, test_slack: bool = False):
    """메인 실행 함수"""
    setup_logging()
    logger = logging.getLogger(__name__)

    if test_slack:
        logger.info("테스트 Slack 메시지 발송")
        sender = SlackSender()
        success = sender.send_test()
        sys.exit(0 if success else 1)

    logger.info("=" * 50)
    logger.info("미국 주식 시장 일일 리포트 생성 시작")
    logger.info("=" * 50)

    try:
        # 1. S&P 500 종목 리스트 가져오기
        logger.info("1. S&P 500 종목 리스트 가져오는 중...")
        tickers = get_all_tickers()
        if not tickers:
            logger.error("S&P 500 종목 리스트를 가져올 수 없습니다.")
            sys.exit(1)
        logger.info(f"   {len(tickers)}개 종목 로드 완료")

        # 2. 주가 데이터 수집
        logger.info("2. 주가 데이터 수집 중...")
        fetcher = StockDataFetcher()
        stock_data = fetcher.fetch_batch(tickers)
        logger.info(f"   {len(stock_data)}개 종목 데이터 수집 완료")

        # 경기 사이클 분석용 추가 티커
        cycle_extra_tickers = ["^TNX", "^IRX", "^VIX", "HYG", "TLT"]
        cycle_data = fetcher.fetch_batch(cycle_extra_tickers, include_spy=False)
        logger.info(f"   사이클 분석용 {len(cycle_data)}개 티커 수집 완료")

        # 시장 요약 (SPY, QQQ 등)
        market_summary = fetcher.get_market_summary()
        logger.info(f"   시장 지수 요약: {list(market_summary.keys())}")

        # 3. 기술적 분석
        logger.info("3. 기술적 분석 수행 중...")
        tech_analyzer = TechnicalAnalyzer()
        # SPY 데이터 추출 (상대강도 계산용)
        spy_df = stock_data.get("SPY")
        analysis_results = tech_analyzer.analyze_batch(stock_data, spy_df)
        logger.info(f"   {len(analysis_results)}개 종목 분석 완료")

        # 4. 섹터별 분석
        logger.info("4. 섹터별 분석 수행 중...")
        sector_analyzer = SectorAnalyzer(stock_data)
        sector_performance = sector_analyzer.analyze_all_sectors()
        market_breadth = sector_analyzer.get_market_breadth()
        top_movers = sector_analyzer.get_top_movers(n=10)
        logger.info(f"   {len(sector_performance)}개 섹터 분석 완료")

        # 4-1. 경기 사이클 분석
        logger.info("4-1. 경기 사이클 분석 중...")
        from analysis.business_cycle import BusinessCycleAnalyzer
        cycle_analyzer = BusinessCycleAnalyzer()
        all_data_for_cycle = {**stock_data, **cycle_data}
        cycle_result = cycle_analyzer.analyze(
            sector_performance=sector_performance,
            market_breadth=market_breadth,
            stock_data=all_data_for_cycle,
        )
        if cycle_result:
            logger.info(f"   경기 국면: {cycle_result.current_phase} ({cycle_result.phase_position:.0f}°)")
        else:
            logger.warning("   경기 사이클 분석 실패 (리포트 생성 계속)")

        # 4-2. 부진 섹터 종목 수집 (추천 제외용)
        lagging_sector_tickers = set()
        if cycle_result and cycle_result.lagging_sectors:
            sector_map = get_sp500_by_sector()
            for sector_name in cycle_result.lagging_sectors:
                tickers_in_sector = sector_map.get(sector_name, [])
                lagging_sector_tickers.update(tickers_in_sector)
            logger.info(f"   부진 섹터 ({', '.join(cycle_result.lagging_sectors)}) 종목 {len(lagging_sector_tickers)}개 추천 제외")

        # 5. 매수 신호 감지
        logger.info("5. 매수 신호 감지 중...")
        signal_detector = SignalDetector(analysis_results, exclude_tickers=lagging_sector_tickers)
        signals = signal_detector.get_all_signals()

        logger.info(f"   RSI 과매도: {len(signals['rsi_oversold'])}개")
        logger.info(f"   MACD 골든크로스: {len(signals['macd_golden_cross'])}개")
        logger.info(f"   1시그마 근접: {len(signals['sigma_reversion'])}개")
        logger.info(f"   복합 신호: {len(signals['combined'])}개")

        # 최종 추천 종목 선정 (가중치 점수 시스템)
        enhanced_rec = signal_detector.get_enhanced_recommendation()
        if enhanced_rec:
            logger.info(f"   오늘의 추천: {enhanced_rec.ticker} (${enhanced_rec.close}, 점수: {enhanced_rec.score}, 신뢰도: {enhanced_rec.confidence})")
            # EnhancedRecommendation을 dict로 변환
            # 템플릿에서 필요한 추가 지표를 analysis_results에서 추출
            rec_analysis = analysis_results.get(enhanced_rec.ticker, {})
            rec_macd = rec_analysis.get("macd")
            rec_bollinger = rec_analysis.get("bollinger")
            rec_atr = rec_analysis.get("atr")
            rec_kalman = rec_analysis.get("kalman")
            rec_obv = rec_analysis.get("obv")
            rec_stochastic = rec_analysis.get("stochastic")
            rec_squeeze = rec_analysis.get("squeeze")

            recommendation = {
                "ticker": enhanced_rec.ticker,
                "score": enhanced_rec.score,
                "confidence": enhanced_rec.confidence,
                "close": enhanced_rec.close,
                "change_pct": enhanced_rec.change_pct,
                "target_price": enhanced_rec.target_price,
                "target_return": round(((enhanced_rec.target_price - enhanced_rec.close) / enhanced_rec.close) * 100, 2) if enhanced_rec.close else None,
                "stop_loss": enhanced_rec.stop_loss,
                "risk_reward_ratio": enhanced_rec.risk_reward_ratio,
                "bullish_factors": enhanced_rec.bullish_factors,
                "warning_factors": enhanced_rec.warning_factors,
                "score_breakdown": {
                    "rsi_score": enhanced_rec.score_breakdown.rsi_score,
                    "volume_score": enhanced_rec.score_breakdown.volume_score,
                    "adx_score": enhanced_rec.score_breakdown.adx_score,
                    "macd_score": enhanced_rec.score_breakdown.macd_score,
                    "bollinger_score": enhanced_rec.score_breakdown.bollinger_score,
                    "relative_strength_score": enhanced_rec.score_breakdown.relative_strength_score,
                    "week52_score": enhanced_rec.score_breakdown.week52_score,
                    "obv_score": enhanced_rec.score_breakdown.obv_score,
                    "stochastic_score": enhanced_rec.score_breakdown.stochastic_score,
                    "squeeze_score": enhanced_rec.score_breakdown.squeeze_score,
                },
                "holding_period": enhanced_rec.holding_period,
                "source": enhanced_rec.source,
                "rsi": enhanced_rec.rsi,
                "adx": enhanced_rec.adx,
                "macd_signal": "골든크로스" if rec_macd and rec_macd.is_bullish_cross else None,
                "bollinger_z_score": round(rec_bollinger.z_score, 2) if rec_bollinger else None,
                "volume_ratio": enhanced_rec.volume_ratio,
                "atr_pct": round(rec_atr.atr / enhanced_rec.close * 100, 2) if rec_atr and enhanced_rec.close else None,
                "relative_strength_20d": enhanced_rec.relative_strength_20d,
                "week52_position": enhanced_rec.week52_position,
                "kalman_predicted_price": round(rec_kalman.predicted_price, 2) if rec_kalman else None,
                "kalman_trend_velocity": round(rec_kalman.trend_velocity, 4) if rec_kalman else None,
                # 새 지표
                "obv_trend": rec_obv.obv_trend if rec_obv else None,
                "stochastic_k": round(rec_stochastic.k, 1) if rec_stochastic else None,
                "squeeze_status": "ON" if rec_squeeze and rec_squeeze.is_squeeze_on else ("OFF" if rec_squeeze else None),
                "squeeze_momentum": rec_squeeze.momentum_direction if rec_squeeze else None,
                "reasons": enhanced_rec.bullish_factors,
                "disclaimer": enhanced_rec.disclaimer,
                "recommendation_method": "Enhanced",
            }
        else:
            # Fallback to legacy recommendation
            recommendation = signal_detector.get_top_recommendation()
            if recommendation:
                logger.info(f"   오늘의 추천 (레거시): {recommendation['ticker']} (${recommendation['close']})")
                # Enhanced 전용 필드 추가 (템플릿 호환성)
                recommendation.setdefault("score", None)
                recommendation.setdefault("confidence", None)
                recommendation.setdefault("bullish_factors", [])
                recommendation.setdefault("warning_factors", [])
                recommendation.setdefault("score_breakdown", None)
                recommendation.setdefault("kalman_predicted_price", None)
                recommendation.setdefault("kalman_trend_velocity", None)
                recommendation.setdefault("obv_trend", None)
                recommendation.setdefault("stochastic_k", None)
                recommendation.setdefault("squeeze_status", None)
                recommendation.setdefault("squeeze_momentum", None)
                recommendation.setdefault("recommendation_method", "Legacy")

        # 5-1. 칼만 필터 추천 종목 선정
        logger.info("5-1. 칼만 필터 추천 종목 선정 중...")
        exclude_tickers = [recommendation["ticker"]] if recommendation else []
        kalman_rec = signal_detector.get_kalman_recommendation(exclude_tickers=exclude_tickers)
        recommendation_kalman = None
        if kalman_rec:
            logger.info(f"   칼만 추천: {kalman_rec.ticker} (${kalman_rec.close}, 점수: {kalman_rec.score}, 칼만예측: ${kalman_rec.kalman_predicted_price:.2f})")
            kal_analysis = analysis_results.get(kalman_rec.ticker, {})
            kal_macd = kal_analysis.get("macd")
            kal_bollinger = kal_analysis.get("bollinger")
            kal_atr = kal_analysis.get("atr")
            kal_kalman = kal_analysis.get("kalman")
            kal_obv = kal_analysis.get("obv")
            kal_stochastic = kal_analysis.get("stochastic")
            kal_squeeze = kal_analysis.get("squeeze")

            recommendation_kalman = {
                "ticker": kalman_rec.ticker,
                "score": kalman_rec.score,
                "confidence": kalman_rec.confidence,
                "close": kalman_rec.close,
                "change_pct": kalman_rec.change_pct,
                "target_price": kalman_rec.target_price,
                "target_return": round(((kalman_rec.target_price - kalman_rec.close) / kalman_rec.close) * 100, 2) if kalman_rec.close else None,
                "stop_loss": kalman_rec.stop_loss,
                "risk_reward_ratio": kalman_rec.risk_reward_ratio,
                "bullish_factors": kalman_rec.bullish_factors,
                "warning_factors": kalman_rec.warning_factors,
                "score_breakdown": {
                    "rsi_score": kalman_rec.score_breakdown.rsi_score,
                    "volume_score": kalman_rec.score_breakdown.volume_score,
                    "adx_score": kalman_rec.score_breakdown.adx_score,
                    "macd_score": kalman_rec.score_breakdown.macd_score,
                    "bollinger_score": kalman_rec.score_breakdown.bollinger_score,
                    "relative_strength_score": kalman_rec.score_breakdown.relative_strength_score,
                    "week52_score": kalman_rec.score_breakdown.week52_score,
                    "obv_score": kalman_rec.score_breakdown.obv_score,
                    "stochastic_score": kalman_rec.score_breakdown.stochastic_score,
                    "squeeze_score": kalman_rec.score_breakdown.squeeze_score,
                },
                "holding_period": kalman_rec.holding_period,
                "source": kalman_rec.source,
                "rsi": kalman_rec.rsi,
                "adx": kalman_rec.adx,
                "macd_signal": "골든크로스" if kal_macd and kal_macd.is_bullish_cross else None,
                "bollinger_z_score": round(kal_bollinger.z_score, 2) if kal_bollinger else None,
                "volume_ratio": kalman_rec.volume_ratio,
                "atr_pct": round(kal_atr.atr / kalman_rec.close * 100, 2) if kal_atr and kalman_rec.close else None,
                "relative_strength_20d": kalman_rec.relative_strength_20d,
                "week52_position": kalman_rec.week52_position,
                "kalman_predicted_price": round(kal_kalman.predicted_price, 2) if kal_kalman else None,
                "kalman_trend_velocity": round(kal_kalman.trend_velocity, 4) if kal_kalman else None,
                "obv_trend": kal_obv.obv_trend if kal_obv else None,
                "stochastic_k": round(kal_stochastic.k, 1) if kal_stochastic else None,
                "squeeze_status": "ON" if kal_squeeze and kal_squeeze.is_squeeze_on else ("OFF" if kal_squeeze else None),
                "squeeze_momentum": kal_squeeze.momentum_direction if kal_squeeze else None,
                "reasons": kalman_rec.bullish_factors,
                "disclaimer": kalman_rec.disclaimer,
                "recommendation_method": "Kalman",
            }
        else:
            logger.info("   칼만 필터 추천 종목 없음")

        # 5-2. 장기 투자 추천 종목 선정
        logger.info("5-2. 장기 투자 추천 종목 선정 중...")
        longterm_recommendations = signal_detector.get_longterm_recommendations()
        if longterm_recommendations:
            logger.info(f"   장기 추천 {len(longterm_recommendations)}개 종목 선정")
        else:
            logger.info("   장기 추천 종목 없음 (하드 필터 통과 종목 부족)")

        # 6. 뉴스 수집
        logger.info("6. 뉴스 수집 중...")
        news_fetcher = NewsFetcher()
        news = news_fetcher.get_all_news(
            top_movers=top_movers,
            sector_performance=sector_performance
        )
        logger.info(f"   {len(news.get('hot_stocks_news', []))}개 핫 종목 뉴스 수집")
        logger.info(f"   {len(news.get('sector_highlights', []))}개 섹터 뉴스 수집")

        # 6-1. 경제 캘린더 수집
        logger.info("6-1. 경제 캘린더 수집 중...")
        economic_cal = EconomicCalendar()
        economic_calendar = economic_cal.get_week_calendar()
        logger.info(f"   경제 캘린더 로드 완료")

        # 6-2. 실적 발표 캘린더 수집
        logger.info("6-2. 실적 발표 캘린더 수집 중...")
        earnings_cal = EarningsCalendar()
        earnings_calendar = earnings_cal.get_week_calendar()
        logger.info(f"   실적 발표 캘린더 로드 완료")

        # 6-3. 추천 종목 뉴스 감성 분석
        try:
            logger.info("6-3. 추천 종목 뉴스 감성 분석 중...")
            sentiment_tickers = []
            if recommendation:
                sentiment_tickers.append(recommendation["ticker"])
            if recommendation_kalman:
                sentiment_tickers.append(recommendation_kalman["ticker"])
            if longterm_recommendations:
                for lt_rec in longterm_recommendations:
                    sentiment_tickers.append(lt_rec["ticker"])

            if sentiment_tickers:
                ticker_news_map = {}
                for t in sentiment_tickers:
                    t_news = news_fetcher.fetch_ticker_news(t)
                    if t_news:
                        ticker_news_map[t] = t_news

                sentiment_analyzer = NewsSentimentAnalyzer()
                sentiment_results = sentiment_analyzer.analyze_multiple_tickers(ticker_news_map)
                logger.info(f"   {len(sentiment_results)}개 종목 감성 분석 완료")

                def _sentiment_to_dict(ts):
                    """TickerSentiment를 dict로 변환"""
                    return {
                        "avg_compound": ts.avg_compound,
                        "gauge_score": ts.gauge_score,
                        "label": ts.label,
                        "positive_count": ts.positive_count,
                        "negative_count": ts.negative_count,
                        "neutral_count": ts.neutral_count,
                        "total_count": ts.total_count,
                        "news_items": [
                            {
                                "title": ni.title,
                                "link": ni.link,
                                "compound": ni.compound,
                                "label": ni.label,
                            }
                            for ni in ts.news_items
                        ],
                    }

                if recommendation and recommendation["ticker"] in sentiment_results:
                    recommendation["sentiment"] = _sentiment_to_dict(
                        sentiment_results[recommendation["ticker"]]
                    )
                if recommendation_kalman and recommendation_kalman["ticker"] in sentiment_results:
                    recommendation_kalman["sentiment"] = _sentiment_to_dict(
                        sentiment_results[recommendation_kalman["ticker"]]
                    )
                if longterm_recommendations:
                    for lt_rec in longterm_recommendations:
                        if lt_rec["ticker"] in sentiment_results:
                            lt_rec["sentiment"] = _sentiment_to_dict(
                                sentiment_results[lt_rec["ticker"]]
                            )
            else:
                logger.info("   감성 분석 대상 종목 없음")
        except Exception as e:
            logger.warning(f"   감성 분석 실패 (리포트 생성 계속): {e}")

        # 7. 리포트 생성
        logger.info("7. 리포트 생성 중...")
        report_gen = ReportGenerator()
        html_report = report_gen.generate(
            market_summary=market_summary,
            market_breadth=market_breadth,
            sector_performance=sector_performance,
            signals=signals,
            top_movers=top_movers,
            news=news,
            recommendation=recommendation,
            recommendation_kalman=recommendation_kalman,
            economic_calendar=economic_calendar,
            earnings_calendar=earnings_calendar,
            longterm_recommendations=longterm_recommendations,
            business_cycle=cycle_result,
        )

        # 리포트 파일 저장
        report_path = report_gen.save_to_file(html_report)
        logger.info(f"   리포트 저장: {report_path}")

        # 7-1. 추천 종목 저장 (프리마켓 리포트용)
        try:
            from data.premarket_tickers import save_recommendations
            rec_tickers = []
            if recommendation:
                rec_tickers.append(recommendation["ticker"])
            if recommendation_kalman:
                rec_tickers.append(recommendation_kalman["ticker"])
            if longterm_recommendations:
                for lt_rec in longterm_recommendations:
                    rec_tickers.append(lt_rec["ticker"])
            if rec_tickers:
                save_recommendations(rec_tickers)
                logger.info(f"   추천 종목 {len(rec_tickers)}개 저장 (프리마켓 용)")
        except Exception as e:
            logger.warning(f"   추천 종목 저장 실패: {e}")

        # 7-2. 추천 종목 성과 추적
        try:
            from tracking import RecommendationRecorder, OutcomeEvaluator, SummaryGenerator

            # 7-2a. 이전 pending 추천 종목 평가 (먼저 실행)
            logger.info("7-2a. 이전 추천 종목 성과 평가 중...")
            evaluator = OutcomeEvaluator()
            evaluated_count = evaluator.evaluate()
            logger.info(f"   {evaluated_count}건 성과 평가 완료")

            # 7-2b. 오늘의 추천 종목 기록
            logger.info("7-2b. 오늘의 추천 종목 기록 중...")
            recorder = RecommendationRecorder()
            recorded_count = recorder.record(
                recommendation=recommendation,
                recommendation_kalman=recommendation_kalman,
                longterm_recommendations=longterm_recommendations,
            )
            logger.info(f"   {recorded_count}건 기록 완료")

            # 7-2c. 통계 요약 생성
            logger.info("7-2c. 통계 요약 생성 중...")
            summary_gen = SummaryGenerator()
            summary_gen.generate()
            logger.info("   통계 요약 생성 완료")
        except Exception as e:
            logger.warning(f"   추천 종목 성과 추적 실패 (리포트 발송 계속): {e}")

        # 8. Slack 발송
        if dry_run:
            logger.info("8. [DRY-RUN] Slack 발송 건너뜀")
            logger.info(f"   생성된 리포트를 확인하세요: {report_path}")
        else:
            logger.info("8. Slack 발송 중...")
            slack_sender = SlackSender()
            success = slack_sender.send(
                html_content=html_report,
                market_summary=market_summary,
                recommendation=recommendation,
                recommendation_kalman=recommendation_kalman,
            )

            if success:
                logger.info("   Slack 발송 완료!")
            else:
                logger.error("   Slack 발송 실패")
                sys.exit(1)

        logger.info("=" * 50)
        logger.info("리포트 생성 완료!")
        logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="미국 주식 시장 일일 리포트 생성"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="발송 없이 리포트만 생성",
    )
    parser.add_argument(
        "--test-slack",
        action="store_true",
        help="테스트 Slack 메시지 발송",
    )

    args = parser.parse_args()
    main(dry_run=args.dry_run, test_slack=args.test_slack)
