from src.validation.schemas import ValidationStatus, IntakeValidationResult, DatasetRegistryEntry

class ValidationContract:
    @staticmethod
    def evaluate_readiness(intake: IntakeValidationResult, metadata: DatasetRegistryEntry) -> ValidationStatus:
        if not intake.files_exist or not intake.readable_media:
            return ValidationStatus.NOT_READY

        if intake.duplicate_or_corrupt_files:
            return ValidationStatus.NOT_READY

        if metadata.access_status not in ["downloaded", "verified", "ready for evaluation"]:
            return ValidationStatus.NOT_READY

        if not intake.license_metadata_present and metadata.license_status != "PUBLIC_OPEN":
             return ValidationStatus.NOT_READY

        # Metric fail-closed logic
        # Never infer metric scale merely because GPS exists.
        # Require explicit reference geometry for metric validation.
        has_metric_anchor = intake.reference_geometry_available and metadata.expected_ground_truth

        if has_metric_anchor:
            return ValidationStatus.READY_FOR_METRIC_VALIDATION
        
        # Fallback to relative
        return ValidationStatus.READY_FOR_RELATIVE_VALIDATION
