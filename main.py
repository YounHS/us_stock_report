#!/usr/bin/env python3
"""
미국 주식 시장 일일 리포트 생성 및 발송

Usage:
    python main.py              # 전체 실행 (리포트 생성 + 이메일 발송)
    python main.py --dry-run    # 이메일 발송 없이 리포트만 생성
    python main.py --test-email # 테스트 이메일 발송
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from config.sp500_tickers import get_all_tickers, get_ticker_info
from data.fetcher import StockDataFetcher
from data.calendar import EconomicCalendar, EarningsCalendar
from analysis.technical import TechnicalAnalyzer
from analysis.sector import SectorAnalyzer
from analysis.signals import SignalDetector
from news.fetcher import NewsFetcher
from report.generator import ReportGenerator
from notification.email_sender import EmailSender


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, settings.general.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(dry_run: bool = False, test_email: bool = False):
    """메인 실행 함수"""
    setup_logging()
    logger = logging.getLogger(__name__)

    if test_email:
        logger.info("테스트 이메일 발송")
        sender = EmailSender()
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

        # 5. 매수 신호 감지
        logger.info("5. 매수 신호 감지 중...")
        signal_detector = SignalDetector(analysis_results)
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
                "reasons": enhanced_rec.bullish_factors,
                "disclaimer": enhanced_rec.disclaimer,
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

        # 5-1. 장기 투자 추천 종목 선정
        logger.info("5-1. 장기 투자 추천 종목 선정 중...")
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
            economic_calendar=economic_calendar,
            earnings_calendar=earnings_calendar,
            longterm_recommendations=longterm_recommendations,
        )

        # 리포트 파일 저장
        report_path = report_gen.save_to_file(html_report)
        logger.info(f"   리포트 저장: {report_path}")

        # 8. 이메일 발송
        if dry_run:
            logger.info("8. [DRY-RUN] 이메일 발송 건너뜀")
            logger.info(f"   생성된 리포트를 확인하세요: {report_path}")
        else:
            logger.info("8. 이메일 발송 중...")
            email_sender = EmailSender()
            success = email_sender.send(html_report)

            if success:
                logger.info("   이메일 발송 완료!")
            else:
                logger.error("   이메일 발송 실패")
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
        help="이메일 발송 없이 리포트만 생성",
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="테스트 이메일 발송",
    )

    args = parser.parse_args()
    main(dry_run=args.dry_run, test_email=args.test_email)
