"""Slack 알림 발송 모듈"""

import logging
import subprocess
import tempfile
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from config.settings import settings

logger = logging.getLogger(__name__)


class SlackSender:
    """Slack Bot Token 기반 메시지 발송기"""

    def __init__(self):
        self.token = settings.slack.bot_token
        self.channel = settings.slack.channel
        self.client = WebClient(token=self.token)

    def _build_summary_text(
        self,
        market_summary: Optional[dict] = None,
        recommendation: Optional[dict] = None,
        recommendation_kalman: Optional[dict] = None,
    ) -> str:
        """Slack 요약 메시지 텍스트 생성"""
        tz = ZoneInfo(settings.general.timezone)
        date_str = datetime.now(tz).strftime("%Y-%m-%d")

        lines = [f"*[주식리포트] {date_str} 미국 주식 시장 일일 리포트*"]

        # 시장 요약
        if market_summary:
            lines.append("")
            lines.append("*시장 요약*")
            for ticker, data in market_summary.items():
                price = data.get("close", 0)
                change = data.get("change_pct", 0)
                arrow = "+" if change >= 0 else ""
                lines.append(f"  {ticker}: ${price:.2f} ({arrow}{change:.2f}%)")

        # 추천 종목
        if recommendation:
            method = recommendation.get("recommendation_method", "")
            method_tag = f" [{method}]" if method else ""
            lines.append("")
            lines.append(f"*오늘의 추천: {recommendation.get('ticker', 'N/A')}*{method_tag}")

            score = recommendation.get("score")
            confidence = recommendation.get("confidence")
            if score is not None:
                lines.append(f"  점수: {score}/100 ({confidence or 'N/A'})")

            close = recommendation.get("close")
            target = recommendation.get("target_price")
            target_ret = recommendation.get("target_return")
            if close is not None:
                target_str = ""
                if target is not None and target_ret is not None:
                    target_str = f" | 목표가: ${target:.2f} ({'+' if target_ret >= 0 else ''}{target_ret:.1f}%)"
                lines.append(f"  현재가: ${close:.2f}{target_str}")

            rsi = recommendation.get("rsi")
            adx = recommendation.get("adx")
            if rsi is not None or adx is not None:
                parts = []
                if rsi is not None:
                    parts.append(f"RSI: {rsi:.1f}")
                if adx is not None:
                    parts.append(f"ADX: {adx:.1f}")
                lines.append(f"  {' | '.join(parts)}")

            sentiment = recommendation.get("sentiment")
            if sentiment:
                lines.append(f"  뉴스 감성: {sentiment['label']} ({sentiment['positive_count']}+ / {sentiment['negative_count']}-)")

        # 칼만 필터 추천 종목
        if recommendation_kalman:
            method = recommendation_kalman.get("recommendation_method", "")
            method_tag = f" [{method}]" if method else ""
            lines.append("")
            lines.append(f"*칼만 필터 추천: {recommendation_kalman.get('ticker', 'N/A')}*{method_tag}")

            kal_score = recommendation_kalman.get("score")
            kal_confidence = recommendation_kalman.get("confidence")
            if kal_score is not None:
                lines.append(f"  점수: {kal_score}/100 ({kal_confidence or 'N/A'})")

            kal_close = recommendation_kalman.get("close")
            kal_target = recommendation_kalman.get("target_price")
            kal_target_ret = recommendation_kalman.get("target_return")
            kal_kalman_price = recommendation_kalman.get("kalman_predicted_price")
            if kal_close is not None:
                target_str = ""
                if kal_target is not None and kal_target_ret is not None:
                    target_str = f" | 목표가: ${kal_target:.2f} ({'+' if kal_target_ret >= 0 else ''}{kal_target_ret:.1f}%)"
                lines.append(f"  현재가: ${kal_close:.2f}{target_str}")
            if kal_kalman_price is not None:
                lines.append(f"  칼만 예측가: ${kal_kalman_price:.2f}")

            kal_sentiment = recommendation_kalman.get("sentiment")
            if kal_sentiment:
                lines.append(f"  뉴스 감성: {kal_sentiment['label']} ({kal_sentiment['positive_count']}+ / {kal_sentiment['negative_count']}-)")

        lines.append("")
        lines.append("상세 리포트는 첨부 파일을 확인하세요.")

        return "\n".join(lines)

    def send(
        self,
        html_content: str,
        market_summary: Optional[dict] = None,
        recommendation: Optional[dict] = None,
        recommendation_kalman: Optional[dict] = None,
    ) -> bool:
        """
        Slack 채널에 요약 메시지 + HTML 리포트 파일 전송

        Args:
            html_content: HTML 리포트 본문
            market_summary: 시장 지수 요약 dict
            recommendation: 추천 종목 dict

        Returns:
            성공 여부
        """
        if not self.token:
            logger.error("SLACK_BOT_TOKEN이 설정되지 않았습니다.")
            return False

        if not self.channel:
            logger.error("SLACK_CHANNEL이 설정되지 않았습니다.")
            return False

        tz = ZoneInfo(settings.general.timezone)
        date_str = datetime.now(tz).strftime("%Y-%m-%d")

        summary_text = self._build_summary_text(market_summary, recommendation, recommendation_kalman)

        try:
            # 1. 요약 메시지 발송
            logger.info(f"Slack 메시지 발송 중: {self.channel}")
            msg_resp = self.client.chat_postMessage(
                channel=self.channel,
                text=summary_text,
                mrkdwn=True,
            )

            # chat_postMessage 응답에서 채널 ID 추출 (files_upload_v2에 필요)
            channel_id = msg_resp["channel"]

            # 2. HTML → PDF 변환 후 업로드
            logger.info("HTML → PDF 변환 중...")
            filename = f"stock_report_{date_str}.pdf"

            tmp_html_fd, tmp_html_path = tempfile.mkstemp(suffix=".html")
            tmp_pdf_path = tmp_html_path.replace(".html", ".pdf")
            try:
                with os.fdopen(tmp_html_fd, "w", encoding="utf-8") as f:
                    f.write(html_content)

                result = subprocess.run(
                    [
                        "google-chrome",
                        "--headless",
                        "--disable-gpu",
                        "--no-sandbox",
                        f"--print-to-pdf={tmp_pdf_path}",
                        tmp_html_path,
                    ],
                    capture_output=True,
                    timeout=30,
                )

                if not os.path.exists(tmp_pdf_path):
                    logger.error(f"PDF 변환 실패: {result.stderr.decode()}")
                    return False

                logger.info("PDF 리포트 파일 업로드 중...")
                self.client.files_upload_v2(
                    channel=channel_id,
                    file=tmp_pdf_path,
                    filename=filename,
                    title=f"{date_str} 미국 주식 시장 일일 리포트",
                    initial_comment="상세 리포트 파일입니다.",
                )
            finally:
                for p in (tmp_html_path, tmp_pdf_path):
                    if os.path.exists(p):
                        os.unlink(p)

            logger.info("Slack 발송 완료")
            return True

        except SlackApiError as e:
            logger.error(f"Slack API 오류: {e.response['error']}")
            return False

        except Exception as e:
            logger.error(f"Slack 발송 실패: {e}")
            return False

    def _build_premarket_summary_text(
        self,
        market_summary: Optional[dict] = None,
        premarket_data: Optional[dict] = None,
        significant_movers: Optional[dict] = None,
        opening_surge_recommendations: Optional[list] = None,
    ) -> str:
        """프리마켓 Slack 요약 메시지 텍스트 생성"""
        tz = ZoneInfo(settings.general.timezone)
        date_str = datetime.now(tz).strftime("%Y-%m-%d")

        lines = [f"*[프리마켓] {date_str} 미국 주식 프리마켓 리포트*"]

        # 시장 지수 프리마켓
        if premarket_data:
            lines.append("")
            lines.append("*시장 지수 프리마켓*")
            for ticker, pm in premarket_data.items():
                if pm.get("has_premarket") and pm.get("pre_market_price"):
                    price = pm["pre_market_price"]
                    change_pct = pm.get("pre_market_change_pct", 0)
                    arrow = "+" if change_pct >= 0 else ""
                    lines.append(f"  {ticker}: ${price:.2f} ({arrow}{change_pct:.2f}%)")
                else:
                    prev = pm.get("regular_previous_close")
                    if prev:
                        lines.append(f"  {ticker}: 전일종가 ${prev:.2f} (PM 없음)")

        # 주요 변동 종목
        if significant_movers:
            gainers = significant_movers.get("gainers", [])
            losers = significant_movers.get("losers", [])
            if gainers:
                lines.append("")
                lines.append("*PM 주요 상승*")
                for g in gainers[:5]:
                    lines.append(f"  {g['ticker']}: +{g['pre_market_change_pct']}%")
            if losers:
                lines.append("")
                lines.append("*PM 주요 하락*")
                for l in losers[:5]:
                    lines.append(f"  {l['ticker']}: {l['pre_market_change_pct']}%")

        # 개장 급등 추천
        if opening_surge_recommendations:
            lines.append("")
            lines.append("*개장 급등 추천 (Opening Surge)*")
            for rec in opening_surge_recommendations:
                ticker = rec.get("ticker", "")
                pm_pct = rec.get("pm_change_pct", 0)
                score = rec.get("score", 0)
                target = rec.get("target_price", 0)
                stop = rec.get("stop_loss", 0)
                lines.append(f"  {ticker}: PM +{pm_pct:.2f}% | 점수: {score} | 목표: ${target:.2f} | 손절: ${stop:.2f}")

        lines.append("")
        lines.append("상세 리포트는 첨부 파일을 확인하세요.")

        return "\n".join(lines)

    def send_premarket(
        self,
        html_content: str,
        market_summary: Optional[dict] = None,
        premarket_data: Optional[dict] = None,
        significant_movers: Optional[dict] = None,
        opening_surge_recommendations: Optional[list] = None,
    ) -> bool:
        """
        프리마켓 리포트 Slack 발송

        Args:
            html_content: HTML 리포트 본문
            market_summary: 시장 지수 요약
            premarket_data: 시장 ETF 프리마켓 데이터
            significant_movers: 주요 변동 종목

        Returns:
            성공 여부
        """
        if not self.token:
            logger.error("SLACK_BOT_TOKEN이 설정되지 않았습니다.")
            return False

        if not self.channel:
            logger.error("SLACK_CHANNEL이 설정되지 않았습니다.")
            return False

        tz = ZoneInfo(settings.general.timezone)
        date_str = datetime.now(tz).strftime("%Y-%m-%d")

        summary_text = self._build_premarket_summary_text(
            market_summary, premarket_data, significant_movers,
            opening_surge_recommendations,
        )

        try:
            # 1. 요약 메시지 발송
            logger.info(f"Slack 프리마켓 메시지 발송 중: {self.channel}")
            msg_resp = self.client.chat_postMessage(
                channel=self.channel,
                text=summary_text,
                mrkdwn=True,
            )

            channel_id = msg_resp["channel"]

            # 2. HTML → PDF 변환 후 업로드
            logger.info("HTML → PDF 변환 중...")
            filename = f"premarket_report_{date_str}.pdf"

            tmp_html_fd, tmp_html_path = tempfile.mkstemp(suffix=".html")
            tmp_pdf_path = tmp_html_path.replace(".html", ".pdf")
            try:
                with os.fdopen(tmp_html_fd, "w", encoding="utf-8") as f:
                    f.write(html_content)

                result = subprocess.run(
                    [
                        "google-chrome",
                        "--headless",
                        "--disable-gpu",
                        "--no-sandbox",
                        f"--print-to-pdf={tmp_pdf_path}",
                        tmp_html_path,
                    ],
                    capture_output=True,
                    timeout=30,
                )

                if not os.path.exists(tmp_pdf_path):
                    logger.error(f"PDF 변환 실패: {result.stderr.decode()}")
                    return False

                logger.info("PDF 프리마켓 리포트 파일 업로드 중...")
                self.client.files_upload_v2(
                    channel=channel_id,
                    file=tmp_pdf_path,
                    filename=filename,
                    title=f"{date_str} 미국 주식 프리마켓 리포트",
                    initial_comment="프리마켓 상세 리포트입니다.",
                )
            finally:
                for p in (tmp_html_path, tmp_pdf_path):
                    if os.path.exists(p):
                        os.unlink(p)

            logger.info("Slack 프리마켓 발송 완료")
            return True

        except SlackApiError as e:
            logger.error(f"Slack API 오류: {e.response['error']}")
            return False

        except Exception as e:
            logger.error(f"Slack 프리마켓 발송 실패: {e}")
            return False

    def send_test(self) -> bool:
        """테스트 메시지 발송"""
        if not self.token:
            logger.error("SLACK_BOT_TOKEN이 설정되지 않았습니다.")
            return False

        if not self.channel:
            logger.error("SLACK_CHANNEL이 설정되지 않았습니다.")
            return False

        try:
            self.client.chat_postMessage(
                channel=self.channel,
                text="*[주식리포트] 테스트 메시지*\nUS Stock Report 시스템에서 발송한 테스트 메시지입니다.\n이 메시지가 정상적으로 수신되었다면 설정이 올바르게 되어 있는 것입니다.",
                mrkdwn=True,
            )
            logger.info("Slack 테스트 메시지 발송 완료")
            return True

        except SlackApiError as e:
            logger.error(f"Slack API 오류: {e.response['error']}")
            return False

        except Exception as e:
            logger.error(f"Slack 테스트 메시지 발송 실패: {e}")
            return False
