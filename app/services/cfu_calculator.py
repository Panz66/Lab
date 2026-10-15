"""
MicroCount - CFU Calculation Module
Colony Forming Units (CFU) calculation with scientific notation formatting.

Formula: CFU/mL = Total Colonies × Dilution Factor / Inoculation Volume (mL)

Implements standard microbiological counting rules:
- APHA (American Public Health Association) guidelines
- USP (US Pharmacopeia) counting standards
- ISO 7218 microbiology standards
"""
import math
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class CFUCalculationResult:
    # Input parameters
    total_colonies: int
    dilution_factor: float
    inoculation_volume_ml: float

    # Results
    cfu_per_ml: float
    cfu_per_g: Optional[float]  # If sample weight provided
    cfu_per_ml_formatted: str
    scientific_notation: str
    log10_cfu: Optional[float]

    # Quality flags
    count_valid: bool
    count_out_of_range: bool
    count_range_message: str

    # Confidence interval (±20% per standard lab uncertainty)
    ci_lower: float
    ci_upper: float
    ci_formatted: str


class CFUCalculator:
    """
    Calculates Colony Forming Units with full laboratory compliance.

    Standard counting ranges:
    - APHA standard plate count: 25–250 colonies per plate
    - FDA BAM: 25–250 colonies per plate
    - ISO 7218: 10–300 colonies per plate
    - USP Pharmacopeia: 30–300 colonies per plate
    """

    # Acceptable count range (ISO 7218 - most permissive laboratory standard)
    MIN_COUNTABLE = 10
    MAX_COUNTABLE = 300

    # Uncertainty factor (±20% expanded uncertainty, ISO GUM)
    UNCERTAINTY_FACTOR = 0.20

    def calculate(
        self,
        total_colonies: int,
        dilution_factor: float,
        inoculation_volume_ml: float,
        sample_weight_g: Optional[float] = None,
    ) -> CFUCalculationResult:
        """
        Calculate CFU/mL from colony count.

        Args:
            total_colonies: Number of colonies counted
            dilution_factor: Sample dilution (e.g., 100 for 10^-2)
            inoculation_volume_ml: Volume plated in mL
            sample_weight_g: Optional sample weight for CFU/g calculation
        """
        if dilution_factor <= 0:
            raise ValueError(f"Dilution factor must be > 0, got {dilution_factor}")
        if inoculation_volume_ml <= 0:
            raise ValueError(f"Inoculation volume must be > 0, got {inoculation_volume_ml}")

        # Core formula: CFU/mL = N × D / V
        cfu_per_ml = (total_colonies * dilution_factor) / inoculation_volume_ml

        # CFU per gram (if weight provided)
        cfu_per_g = None
        if sample_weight_g and sample_weight_g > 0:
            cfu_per_g = cfu_per_ml / sample_weight_g

        # Formatting
        cfu_formatted = self._format_cfu(cfu_per_ml)
        sci_notation = self._to_scientific(cfu_per_ml)

        # Log10 value
        log10_cfu = math.log10(cfu_per_ml) if cfu_per_ml > 0 else None

        # Confidence interval
        ci_lower = cfu_per_ml * (1 - self.UNCERTAINTY_FACTOR)
        ci_upper = cfu_per_ml * (1 + self.UNCERTAINTY_FACTOR)
        ci_fmt = f"{self._format_cfu(ci_lower)} – {self._format_cfu(ci_upper)}"

        # Validity check
        valid, out_of_range, range_msg = self._check_count_validity(total_colonies)

        logger.info(
            f"CFU calculation: colonies={total_colonies}, DF={dilution_factor}, "
            f"vol={inoculation_volume_ml}mL → {cfu_formatted} CFU/mL"
        )

        return CFUCalculationResult(
            total_colonies=total_colonies,
            dilution_factor=dilution_factor,
            inoculation_volume_ml=inoculation_volume_ml,
            cfu_per_ml=cfu_per_ml,
            cfu_per_g=cfu_per_g,
            cfu_per_ml_formatted=cfu_formatted,
            scientific_notation=sci_notation,
            log10_cfu=round(log10_cfu, 2) if log10_cfu else None,
            count_valid=valid,
            count_out_of_range=out_of_range,
            count_range_message=range_msg,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_formatted=ci_fmt,
        )

    def _format_cfu(self, cfu: float) -> str:
        """Format CFU/mL value with appropriate units and precision."""
        if cfu == 0:
            return "0 CFU/mL"
        elif cfu < 1:
            return f"{cfu:.2e} CFU/mL"
        elif cfu < 1_000:
            return f"{cfu:.0f} CFU/mL"
        elif cfu < 1_000_000:
            return f"{cfu/1_000:.2f} × 10³ CFU/mL"
        elif cfu < 1_000_000_000:
            return f"{cfu/1_000_000:.2f} × 10⁶ CFU/mL"
        else:
            return f"{cfu:.3e} CFU/mL"

    def _to_scientific(self, value: float) -> str:
        """Convert to standard scientific notation string."""
        if value == 0:
            return "0"
        try:
            exp = math.floor(math.log10(abs(value)))
            mantissa = value / (10 ** exp)
            return f"{mantissa:.2f} × 10^{exp}"
        except (ValueError, OverflowError):
            return str(value)

    def _check_count_validity(self, count: int) -> tuple[bool, bool, str]:
        """
        Validate colony count against laboratory standards.

        Returns:
            (valid, out_of_range, message)
        """
        if count == 0:
            return False, True, "No colonies detected. Results may indicate sterility or imaging failure."

        if count < self.MIN_COUNTABLE:
            return True, True, (
                f"Low count ({count} colonies). Below recommended minimum of {self.MIN_COUNTABLE}. "
                "Consider re-analyzing with lower dilution or larger inoculation volume."
            )

        if count > self.MAX_COUNTABLE:
            return True, True, (
                f"High count ({count} colonies). Exceeds recommended maximum of {self.MAX_COUNTABLE}. "
                "Consider re-analyzing with higher dilution factor. "
                "Results reported as TNTC (Too Numerous To Count) threshold exceeded."
            )

        return True, False, (
            f"Colony count ({count}) is within acceptable range "
            f"({self.MIN_COUNTABLE}–{self.MAX_COUNTABLE} colonies per plate)."
        )

    def calculate_dilution_series(
        self,
        colony_counts: list[int],
        dilution_factors: list[float],
        inoculation_volume_ml: float,
    ) -> dict:
        """
        Calculate CFU from multiple dilution series and return best estimate.
        Follows APHA guidelines for selecting the most valid count.
        """
        if len(colony_counts) != len(dilution_factors):
            raise ValueError("colony_counts and dilution_factors must have same length")

        results = []
        best_result = None
        best_count = None

        for count, df in zip(colony_counts, dilution_factors):
            res = self.calculate(count, df, inoculation_volume_ml)
            results.append({
                "dilution_factor": df,
                "colony_count": count,
                "cfu_per_ml": res.cfu_per_ml,
                "cfu_formatted": res.cfu_per_ml_formatted,
                "valid": res.count_valid,
                "in_range": not res.count_out_of_range,
            })

            # Prefer counts in valid range
            if not res.count_out_of_range:
                if best_count is None or abs(count - 150) < abs(best_count - 150):
                    best_count = count
                    best_result = res

        # If no in-range count, use count closest to midpoint
        if best_result is None and results:
            midpoint = (self.MIN_COUNTABLE + self.MAX_COUNTABLE) / 2
            best_idx = min(range(len(colony_counts)), key=lambda i: abs(colony_counts[i] - midpoint))
            best_result = self.calculate(
                colony_counts[best_idx],
                dilution_factors[best_idx],
                inoculation_volume_ml,
            )

        return {
            "series_results": results,
            "best_estimate": best_result,
            "recommended_cfu": best_result.cfu_per_ml if best_result else 0,
        }
