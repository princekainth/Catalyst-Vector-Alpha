# ccn_monitor_mock.py (or directly in your main agent file for now)

class MockCCNMonitor:
    """
    A mock implementation of the CCN Monitor interface for testing purposes.
    Simulates fetching concurrent process states.
    """
    def get_current_process_state(self) -> list[dict]:
        """
        Returns a mock list of dictionaries, each representing a process state
        within the simulated CCN environment.
        """
        mock_data = [
            {
                "process_id": "proc_A_123",
                "process_name": "DataIngestionService",
                "status": "running",
                "start_time": "2025-07-30T16:40:00Z",
                "resources_in_use": ["database_connection_pool_1", "message_queue_input_1"],
                "dependencies": [],
                "last_activity_time": "2025-07-30T16:45:20Z",
                "metadata": {"data_source": "SensorStreamA", "expected_volume_gb": 10},
            },
            {
                "process_id": "proc_B_456",
                "process_name": "AnomalyDetectionEngine",
                "status": "waiting",
                "start_time": "2025-07-30T16:41:00Z",
                "resources_in_use": ["ml_inference_engine_1", "database_connection_pool_1"],
                "dependencies": ["proc_A_123"], # Depends on DataIngestionService
                "last_activity_time": "2025-07-30T16:41:00Z", # Stalled since start?
                "metadata": {"model_version": "v2.1", "alert_threshold": 0.95},
            },
            {
                "process_id": "proc_C_789",
                "process_name": "ResourceOptimizer",
                "status": "running",
                "start_time": "2025-07-30T16:42:00Z",
                "resources_in_use": ["system_scheduler"],
                "dependencies": [],
                "last_activity_time": "2025-07-30T16:45:30Z",
                "metadata": {"optimization_strategy": "cost_efficiency"},
            },
            # Add more mock processes here to simulate different scenarios
            # For example, another process using 'database_connection_pool_1' to simulate contention
            {
                "process_id": "proc_D_000",
                "process_name": "ReportGenerationService",
                "status": "running",
                "start_time": "2025-07-30T16:43:00Z",
                "resources_in_use": ["database_connection_pool_1", "report_output_storage"], # Contention here
                "dependencies": [],
                "last_activity_time": "2025-07-30T16:45:15Z",
                "metadata": {"report_type": "daily_summary"},
            },
        ]
        return mock_data