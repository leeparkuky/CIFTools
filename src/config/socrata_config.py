import os
from dataclasses import dataclass
from dotenv import load_dotenv
from sodapy import Socrata

# Load environment variables
load_dotenv("/root/CIFTools/.env")

@dataclass
class SocrataConfig:
    domain: str = os.getenv("SOCRATA_DOMAIN", "chronicdata.cdc.gov")
    app_token: str = os.getenv("SOCRATA_APP_TOKEN", None)
    user_name: str = os.getenv("SOCRATA_USER_NAME", None)
    password: str = os.getenv("SOCRATA_PASSWORD", None)

    @property
    def client(self):
        """Returns an authenticated Socrata API client."""
        if not hasattr(self, '_client'):
            if self.user_name and self.password:
                self._client = Socrata(self.domain, self.app_token, username=self.user_name, password=self.password)
            else:
                self._client = Socrata(self.domain, self.app_token)
        return self._client
