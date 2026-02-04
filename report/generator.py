"""리포트 생성기"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
import logging

from jinja2 import Environment, FileSystemLoader

from config.settings import settings
from analysis.signals import BuySignal
from analysis.sector import SectorPerformance

logger = logging.getLogger(__name__)


class ReportGenerator:
    """HTML 리포트 생성기"""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True,
        )

    def generate(
        self,
        market_summary: Dict,
        market_breadth: Dict,
        sector_performance: List[SectorPerformance],
        signals: Dict[str, List[BuySignal]],
        top_movers: Dict,
        news: Dict,
        recommendation: Optional[Dict] = None,
        economic_calendar: Optional[Dict] = None,
        earnings_calendar: Optional[Dict] = None,
        report_date: Optional[str] = None,
    ) -> str:
        """
        일일 리포트 HTML 생성

        Args:
            market_summary: 시장 지수 요약
            market_breadth: 상승/하락 종목 수
            sector_performance: 섹터별 성과 리스트
            signals: 매수 신호 딕셔너리
            top_movers: 상위 상승/하락 종목
            news: 뉴스 데이터
            report_date: 리포트 기준 날짜

        Returns:
            HTML 문자열
        """
        template = self.env.get_template("daily_report.html")

        # dataclass를 dict로 변환
        sector_perf_dicts = [
            {
                "name": s.name,
                "daily_return": s.daily_return,
                "weekly_return": s.weekly_return,
                "monthly_return": s.monthly_return,
                "advancing": s.advancing,
                "declining": s.declining,
                "strength": s.strength,
            }
            for s in sector_performance
        ]

        # BuySignal도 dict로 변환
        signals_dicts = {}
        for signal_type, signal_list in signals.items():
            signals_dicts[signal_type] = [
                {
                    "ticker": s.ticker,
                    "close": s.close,
                    "change_pct": s.change_pct,
                    "signal_type": s.signal_type,
                    "signal_strength": s.signal_strength,
                    "details": s.details,
                }
                for s in signal_list
            ]

        context = {
            "report_date": report_date or datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_summary": market_summary,
            "market_breadth": market_breadth,
            "sector_performance": sector_perf_dicts,
            "signals": signals_dicts,
            "top_movers": top_movers,
            "news": news,
            "recommendation": recommendation,
            "economic_calendar": economic_calendar,
            "earnings_calendar": earnings_calendar,
        }

        return template.render(**context)

    def save_to_file(self, html: str, filepath: Optional[str] = None) -> str:
        """
        리포트를 파일로 저장

        Args:
            html: HTML 문자열
            filepath: 저장 경로 (없으면 자동 생성)

        Returns:
            저장된 파일 경로
        """
        if filepath is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"report_{date_str}.html"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"리포트 저장 완료: {filepath}")
        return filepath
