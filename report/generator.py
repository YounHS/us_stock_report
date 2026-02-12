"""리포트 생성기"""

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
from zoneinfo import ZoneInfo
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
        recommendation_kalman: Optional[Dict] = None,
        economic_calendar: Optional[Dict] = None,
        earnings_calendar: Optional[Dict] = None,
        longterm_recommendations: Optional[List[Dict]] = None,
        business_cycle=None,
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

        # BusinessCycleResult → dict 변환 + SVG 마커 좌표 사전 계산
        business_cycle_dict = None
        if business_cycle is not None:
            from analysis.business_cycle import PHASE_LABELS, PHASE_ANGLES
            angle_svg = business_cycle.phase_position - 90  # SVG 좌표계 변환 (12시=0도 → 3시=0도)
            angle_rad = math.radians(angle_svg)
            marker_x = 200 + 130 * math.cos(angle_rad)
            marker_y = 200 + 130 * math.sin(angle_rad)

            business_cycle_dict = {
                "current_phase": business_cycle.current_phase,
                "current_phase_label": PHASE_LABELS.get(business_cycle.current_phase, ""),
                "phase_position": business_cycle.phase_position,
                "marker_x": round(marker_x, 1),
                "marker_y": round(marker_y, 1),
                "phase_probabilities": business_cycle.phase_probabilities,
                "leading_sectors": business_cycle.leading_sectors,
                "lagging_sectors": business_cycle.lagging_sectors,
                "summary": business_cycle.summary,
                "factor_readings": [
                    {
                        "name": f.name,
                        "value": f.value,
                        "display_value": f.display_value,
                        "phase_signal": f.phase_signal,
                        "phase_signal_label": PHASE_LABELS.get(f.phase_signal, "N/A") if f.phase_signal else "N/A",
                        "weight": f.weight,
                        "description": f.description,
                    }
                    for f in business_cycle.factor_readings
                ],
                "phase_labels": PHASE_LABELS,
                "phase_angles": PHASE_ANGLES,
            }

        tz = ZoneInfo(settings.general.timezone)
        now = datetime.now(tz)
        context = {
            "report_date": report_date or now.strftime("%Y-%m-%d"),
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "market_summary": market_summary,
            "market_breadth": market_breadth,
            "sector_performance": sector_perf_dicts,
            "signals": signals_dicts,
            "top_movers": top_movers,
            "news": news,
            "recommendation": recommendation,
            "recommendation_kalman": recommendation_kalman,
            "economic_calendar": economic_calendar,
            "earnings_calendar": earnings_calendar,
            "longterm_recommendations": longterm_recommendations,
            "business_cycle": business_cycle_dict,
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
            tz = ZoneInfo(settings.general.timezone)
            date_str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
            filepath = f"report_{date_str}.html"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"리포트 저장 완료: {filepath}")
        return filepath
