"""
Supervisor Module - Crash Detection & Auto-Recovery
Wraps the cognitive loop with crash protection and restart logic
"""
import time
import traceback
from datetime import datetime
from typing import Callable, Optional
import logging

class CognitiveSupervisor:
    """Monitors and restarts the cognitive loop on crashes"""
    
    def __init__(self, cva_instance, database=None, logger=None):
        self.cva = cva_instance
        self.database = database
        self.logger = logger or logging.getLogger(__name__)
        
        # Crash tracking
        self.crash_count = 0
        self.crash_history = []
        self.max_crashes = 10  # Stop after 10 crashes in quick succession
        self.crash_window = 300  # 5 minutes
        
        # Backoff configuration
        self.base_backoff = 5  # Start with 5 second delay
        self.max_backoff = 300  # Cap at 5 minutes
        
    def _log_crash(self, exception: Exception, loop_cycle_count: int):
        """Log crash to database and console"""
        crash_time = datetime.now()
        crash_info = {
            "timestamp": crash_time.isoformat(),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
            "loop_cycle_count": loop_cycle_count,
            "crash_number": self.crash_count,
        }
        
        # Add to history
        self.crash_history.append(crash_info)
        
        # Log to console
        self.logger.error(f"🔴 COGNITIVE LOOP CRASHED (#{self.crash_count})")
        self.logger.error(f"   Type: {crash_info['exception_type']}")
        self.logger.error(f"   Message: {crash_info['exception_message']}")
        self.logger.error(f"   Cycle: {loop_cycle_count}")
        
        # Log to database if available
        if self.database:
            try:
                # You can add a crashes table later
                self.logger.info("   (Database logging not yet implemented)")
            except Exception as e:
                self.logger.error(f"   Failed to log crash to database: {e}")
        
        return crash_info
    
    def _calculate_backoff(self) -> int:
        """Calculate exponential backoff delay"""
        backoff = min(self.base_backoff * (2 ** (self.crash_count - 1)), self.max_backoff)
        return int(backoff)
    
    def _cleanup_old_crashes(self):
        """Remove crashes older than crash_window"""
        now = datetime.now()
        cutoff = now.timestamp() - self.crash_window
        
        self.crash_history = [
            crash for crash in self.crash_history
            if datetime.fromisoformat(crash["timestamp"]).timestamp() > cutoff
        ]
        
        # Update crash count based on recent history
        self.crash_count = len(self.crash_history)
    
    def _should_stop(self) -> bool:
        """Check if we should stop trying to restart"""
        self._cleanup_old_crashes()
        
        if self.crash_count >= self.max_crashes:
            self.logger.critical(
                f"🛑 TOO MANY CRASHES: {self.crash_count} crashes in {self.crash_window}s"
            )
            self.logger.critical("   Stopping auto-restart to prevent infinite loop")
            return True
        
        return False
    
    def run_supervised(self, tick_sleep: int = 10):
        """
        Run the cognitive loop with supervision
        Automatically restarts on crashes with exponential backoff
        """
        print("🛡️  Supervisor: Starting supervised cognitive loop")
        self.logger.info("🛡️  Supervisor: Starting supervised cognitive loop")
        
        while self.cva.is_running:
            try:
                # Check if we should stop
                if self._should_stop():
                    break
                
                # Run the cognitive loop
                print("✅ Supervisor: Cognitive loop running normally")
                self.logger.info("✅ Supervisor: Cognitive loop running normally")
                self.cva.run_cognitive_loop(tick_sleep=tick_sleep)
                
                # If loop exits cleanly, we're done
                self.logger.info("✅ Supervisor: Cognitive loop exited cleanly")
                break
                
            except KeyboardInterrupt:
                self.logger.info("⚠️  Supervisor: Keyboard interrupt - shutting down")
                self.cva.is_running = False
                break
                
            except Exception as e:
                # Cognitive loop crashed!
                self.crash_count += 1
                loop_count = self.cva.swarm_state.get("cycle_count", 0)
                
                crash_info = self._log_crash(e, loop_count)
                
                # Calculate backoff
                backoff = self._calculate_backoff()
                
                self.logger.warning(f"⏳ Supervisor: Waiting {backoff}s before restart...")
                time.sleep(backoff)
                
                # Try to restart
                if self.cva.is_running:
                    self.logger.info(f"🔄 Supervisor: Attempting restart (attempt #{self.crash_count})")
                else:
                    self.logger.info("🛑 Supervisor: System stopped, not restarting")
                    break
        
        self.logger.info("🏁 Supervisor: Shutdown complete")
        
        # Print crash summary
        if self.crash_history:
            self.logger.info(f"\n📊 Crash Summary: {len(self.crash_history)} crashes occurred")
            for crash in self.crash_history[-5:]:  # Show last 5
                self.logger.info(f"   - {crash['timestamp']}: {crash['exception_type']}")
