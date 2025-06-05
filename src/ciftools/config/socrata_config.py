import os
from dataclasses import dataclass
from dotenv import load_dotenv
from sodapy import Socrata

# Load environment variables
# Allow overriding the .env path with CIFTOOLS_ENV_PATH, default to .env
dotenv_path = os.getenv("CIFTOOLS_ENV_PATH", ".env")
load_dotenv(dotenv_path)


@dataclass
class SocrataConfig:
    domain: str = os.getenv("SOCRATA_DOMAIN", "chronicdata.cdc.gov")
    app_token: str = os.getenv("SOCRATA_APP_TOKEN", None)
    user_name: str = os.getenv("SOCRATA_USER_NAME", None)
    password: str = os.getenv("SOCRATA_PASSWORD", None)

    @property
    def client(self):
        """Returns an authenticated Socrata API client."""
        if not hasattr(self, "_client"):
            if self.user_name and self.password:
                self._client = Socrata(
                    self.domain,
                    self.app_token,
                    username=self.user_name,
                    password=self.password,
                )
            else:
                self._client = Socrata(self.domain, self.app_token)
        return self._client
