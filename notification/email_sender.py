"""이메일 발송 모듈"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class EmailSender:
    """SMTP 이메일 발송기"""

    def __init__(self):
        self.host = settings.smtp.host
        self.port = settings.smtp.port
        self.user = settings.smtp.user
        self.password = settings.smtp.password
        self.from_address = settings.email.from_address or self.user
        self.recipients = settings.email.recipient_list
        self.subject_prefix = settings.email.subject_prefix

    def send(
        self,
        html_content: str,
        subject: Optional[str] = None,
        recipients: Optional[List[str]] = None,
    ) -> bool:
        """
        HTML 이메일 발송

        Args:
            html_content: HTML 본문
            subject: 이메일 제목 (없으면 자동 생성)
            recipients: 수신자 목록 (없으면 설정값 사용)

        Returns:
            성공 여부
        """
        recipients = recipients or self.recipients
        if not recipients:
            logger.error("수신자 목록이 비어있습니다.")
            return False

        if not subject:
            tz = ZoneInfo(settings.general.timezone)
            date_str = datetime.now(tz).strftime("%Y-%m-%d")
            subject = f"{self.subject_prefix} {date_str} 미국 주식 시장 일일 리포트"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_address
        msg["To"] = ", ".join(recipients)

        # HTML 본문 추가
        html_part = MIMEText(html_content, "html", "utf-8")
        msg.attach(html_part)

        try:
            logger.info(f"이메일 발송 시작: {recipients}")

            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.sendmail(self.from_address, recipients, msg.as_string())

            logger.info("이메일 발송 완료")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP 인증 실패: {e}")
            logger.error("Gmail 사용 시 앱 비밀번호를 사용해야 합니다.")
            return False

        except smtplib.SMTPException as e:
            logger.error(f"SMTP 오류: {e}")
            return False

        except Exception as e:
            logger.error(f"이메일 발송 실패: {e}")
            return False

    def send_test(self) -> bool:
        """테스트 이메일 발송"""
        test_html = """
        <html>
        <body>
            <h1>테스트 이메일</h1>
            <p>US Stock Report 시스템에서 발송한 테스트 이메일입니다.</p>
            <p>이 이메일이 정상적으로 수신되었다면 설정이 올바르게 되어 있는 것입니다.</p>
        </body>
        </html>
        """
        return self.send(
            html_content=test_html,
            subject=f"{self.subject_prefix} 테스트 이메일",
        )
