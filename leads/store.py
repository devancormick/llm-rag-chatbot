import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config


class LeadStore:
    """Stores lead information (name, email, company) for chatbot visitors."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or config.LEADS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.leads_file = self.base_dir / "leads.json"

    def _load_leads(self) -> List[dict]:
        if not self.leads_file.exists():
            return []
        try:
            data = self.leads_file.read_text(encoding="utf-8")
            return json.loads(data)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_leads(self, leads: List[dict]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.leads_file.write_text(
            json.dumps(leads, indent=2, default=str),
            encoding="utf-8",
        )

    def add(self, email: str, name: Optional[str] = None, company: Optional[str] = None) -> str:
        """Add a lead. Returns lead_id."""
        leads = self._load_leads()
        lead_id = str(uuid.uuid4())

        existing = next((lead for lead in leads if lead.get("email", "").lower() == email.lower()), None)
        if existing:
            return existing["id"]

        lead = {
            "id": lead_id,
            "email": email.strip(),
            "name": (name or "").strip(),
            "company": (company or "").strip(),
            "created_at": datetime.utcnow().isoformat(),
        }
        leads.append(lead)
        self._save_leads(leads)
        return lead_id

    def get_all(self) -> List[dict]:
        """Return all leads."""
        return self._load_leads()

    def export_csv(self) -> str:
        """Export leads as CSV string."""
        leads = self._load_leads()
        if not leads:
            return "email,name,company,created_at\n"
        headers = ["email", "name", "company", "created_at"]
        lines = [",".join(str(lead.get(h, "")) for h in headers) for lead in leads]
        return "email,name,company,created_at\n" + "\n".join(lines)
