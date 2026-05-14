import os
import shutil
import logging
from typing import Dict, Any, Optional, List

log = logging.getLogger("DesktopHands")

class DesktopHands:
    """
    Actuation layer for CVA Desktop Operator.
    Provides safe, gated filesystem operations.
    """
    
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.allowed_roots = [
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Downloads"),
            self.workspace_root,
            os.path.join(self.workspace_root, "scratch")
        ]
        self.blocked_keywords = [
            ".ssh", ".config", ".gnupg", ".aws", ".kube",
            "/etc", "/usr", "/var", "/root", "/bin", "/lib",
            ".env", "credentials", "tokens", "secrets", "private_keys", ".pem", ".key"
        ]

    def _validate_path(self, path: str) -> bool:
        """Strict path validation against allowlist and blocklist."""
        try:
            # Resolve absolute path and handle symlinks
            abs_path = os.path.realpath(os.path.expanduser(path))
            
            # 1. Block traversal
            if ".." in path:
                log.warning(f"Safety violation: Path traversal attempt: {path}")
                return False
            
            # 2. Block keywords
            path_lower = abs_path.lower()
            for kw in self.blocked_keywords:
                if kw in path_lower:
                    log.warning(f"Safety violation: Blocked keyword '{kw}' in path: {abs_path}")
                    return False
            
            # 3. Check allowlist
            is_allowed = any(abs_path.startswith(root) for root in self.allowed_roots)
            if not is_allowed:
                log.warning(f"Safety violation: Path outside allowed roots: {abs_path}")
                return False
                
            return True
        except Exception as e:
            log.error(f"Path validation exception: {e}")
            return False

    def create_folder(self, path: str) -> Dict[str, Any]:
        if not self._validate_path(path):
            return {"status": "error", "error": "Path validation failed."}
        
        try:
            os.makedirs(path, exist_ok=True)
            return {"status": "ok", "summary": f"Folder created: {path}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def write_text_file(self, path: str, content: str) -> Dict[str, Any]:
        if not self._validate_path(path):
            return {"status": "error", "error": "Path validation failed."}
        
        # Size cap: 1MB
        if len(content) > 1024 * 1024:
            return {"status": "error", "error": "File size exceeds 1MB limit."}

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return {"status": "ok", "summary": f"File written: {path}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        if not self._validate_path(source) or not self._validate_path(destination):
            return {"status": "error", "error": "Path validation failed."}
        
        try:
            shutil.move(source, destination)
            return {"status": "ok", "summary": f"Moved {source} to {destination}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def delete_file(self, path: str) -> Dict[str, Any]:
        if not self._validate_path(path):
            return {"status": "error", "error": "Path validation failed."}
        
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return {"status": "ok", "summary": f"Deleted: {path}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def patch_text_file(self, path: str, search: str, replace: str) -> Dict[str, Any]:
        if not self._validate_path(path):
            return {"status": "error", "error": "Path validation failed."}
        
        try:
            with open(path, "r") as f:
                content = f.read()
            
            if search not in content:
                return {"status": "error", "error": f"Search string not found in {path}"}
            
            new_content = content.replace(search, replace)
            with open(path, "w") as f:
                f.write(new_content)
                
            return {"status": "ok", "summary": f"Patched {path}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
