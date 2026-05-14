import os
import yaml
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

log = logging.getLogger("CapabilityLoader")

@dataclass
class SkillCapability:
    name: str
    description: str
    version: str
    instructions: str
    requirements: Dict[str, Any]
    path: str

class CapabilityLoader:
    def __init__(self, capabilities_dir: str = "capabilities"):
        self.capabilities_dir = capabilities_dir
        self.skills: Dict[str, SkillCapability] = {}

    def load_all(self) -> List[SkillCapability]:
        if not os.path.exists(self.capabilities_dir):
            log.warning(f"Capabilities directory not found: {self.capabilities_dir}")
            return []

        loaded = []
        for root, dirs, files in os.walk(self.capabilities_dir):
            if "SKILL.md" in files:
                skill_path = os.path.join(root, "SKILL.md")
                skill = self.parse_skill_file(skill_path)
                if skill:
                    self.skills[skill.name] = skill
                    loaded.append(skill)
                    log.info(f"Loaded skill: {skill.name} v{skill.version}")
        
        return loaded

    def parse_skill_file(self, file_path: str) -> Optional[SkillCapability]:
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # --- SAFETY VALIDATION ---
            dangerous_patterns = [
                "ignore previous instructions",
                "bypass approval",
                "disable safety",
                "run arbitrary shell",
                "read secrets",
                "exfiltrate",
                "sudo",
                "~/.ssh",
                ".env"
            ]
            
            content_lower = content.lower()
            for pattern in dangerous_patterns:
                if pattern in content_lower:
                    log.error(f"⚠️ SAFETY REJECTION: Skill at {file_path} contains dangerous directive: '{pattern}'")
                    return None

            if not content.startswith("---"):
                log.warning(f"Invalid SKILL.md at {file_path}: Missing frontmatter")
                return None

            parts = content.split("---", 2)
            if len(parts) < 3:
                log.warning(f"Invalid SKILL.md at {file_path}: Incomplete frontmatter")
                return None

            frontmatter_raw = parts[1]
            instructions = parts[2].strip()

            metadata = yaml.safe_load(frontmatter_raw)
            
            return SkillCapability(
                name=metadata.get("name", "unknown"),
                description=metadata.get("description", ""),
                version=str(metadata.get("version", "1.0.0")),
                instructions=instructions,
                requirements=metadata.get("metadata", {}).get("openclaw", {}).get("requires", {}),
                path=file_path
            )
        except Exception as e:
            log.error(f"Failed to parse skill at {file_path}: {e}")
            return None

    def get_system_prompt_fragment(self) -> str:
        """Returns a combined string of all skill instructions to be injected into the system prompt."""
        if not self.skills:
            return ""

        fragments = ["\n## ADAPTIVE CAPABILITIES (SKILLS)\n"]
        for skill in self.skills.values():
            fragments.append(f"### Skill: {skill.name}")
            fragments.append(f"Description: {skill.description}")
            fragments.append(f"Instructions:\n{skill.instructions}\n")
        
        return "\n".join(fragments)
