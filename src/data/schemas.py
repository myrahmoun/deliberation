"""Pydantic schemas for persona bank data structures."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Demographics(BaseModel):
    """Demographic information for a persona."""

    age: Optional[str] = Field(None, description="Age category (e.g., '18-29', '30-49', '50-64', '65+')")
    gender: Optional[str] = Field(None, description="Gender")
    education: Optional[str] = Field(None, description="Education level")
    income: Optional[str] = Field(None, description="Income tier")
    race: Optional[str] = Field(None, description="Race/ethnicity")
    region: Optional[str] = Field(None, description="Census region (Northeast, Midwest, South, West)")
    metro: Optional[str] = Field(None, description="Metro area indicator")
    marital_status: Optional[str] = Field(None, description="Marital status")
    religion: Optional[str] = Field(None, description="Religious affiliation")
    religion_attendance: Optional[str] = Field(None, description="Religious service attendance frequency")


class Ideology(BaseModel):
    """Political ideology information for a persona."""

    party: Optional[str] = Field(None, description="Political party affiliation")
    party_lean: Optional[str] = Field(None, description="Party leaning for independents")
    party_summary: Optional[str] = Field(None, description="Combined party/lean summary")
    ideology: Optional[str] = Field(None, description="Political ideology (e.g., 'Very conservative' to 'Very liberal')")


class OpinionRecord(BaseModel):
    """A single opinion response from a survey."""

    question_id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="Full question text")
    survey_id: str = Field(..., description="Source survey wave identifier")
    response_value: int = Field(..., description="Numeric response code")
    response_label: str = Field(..., description="Human-readable response label")


class Persona(BaseModel):
    """A persona representing one survey respondent."""

    persona_id: str = Field(..., description="Unique persona identifier (survey_respondent)")
    source_survey: str = Field(..., description="Source survey wave (e.g., 'W26')")
    source_respondent_id: str = Field(..., description="Original respondent ID (QKEY)")
    weight: float = Field(1.0, description="Survey weight for this respondent")
    demographics: Demographics = Field(default_factory=Demographics)
    ideology: Ideology = Field(default_factory=Ideology)
    past_opinions: Dict[str, OpinionRecord] = Field(
        default_factory=dict, description="Dict of question_id -> OpinionRecord"
    )


class Question(BaseModel):
    """A survey question with its response options."""

    question_id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="Full question text")
    survey_id: str = Field(..., description="Source survey wave identifier")
    response_options: Dict[int, str] = Field(
        default_factory=dict, description="Mapping of response codes to labels"
    )


class SurveyMetadata(BaseModel):
    """Metadata about a processed survey."""

    survey_id: str = Field(..., description="Survey wave identifier")
    survey_name: str = Field(..., description="Full survey name")
    respondent_count: int = Field(..., description="Number of respondents in survey")
    question_count: int = Field(..., description="Number of opinion questions extracted")
    source_file: str = Field(..., description="Original source file path")


class PersonaBank(BaseModel):
    """Complete persona bank with all personas and metadata."""

    version: str = Field("1.0.0", description="Schema version")
    created_at: datetime = Field(default_factory=datetime.now)
    persona_count: int = Field(0, description="Total number of personas")
    personas: List[Persona] = Field(default_factory=list)
    questions: Dict[str, Question] = Field(
        default_factory=dict, description="Dict of question_id -> Question"
    )
    survey_metadata: List[SurveyMetadata] = Field(default_factory=list)


# =============================================================================
# Habermas Machine Schemas
# =============================================================================


class Issue(BaseModel):
    """A deliberation issue/question from Habermas Machine."""

    issue_id: str = Field(..., description="Unique issue identifier")
    issue_text: str = Field(..., description="The issue question text")
    topic_id: Optional[int] = Field(None, description="Topic category ID")
    affirming_statement: Optional[str] = Field(None, description="Pro-position framing")
    negating_statement: Optional[str] = Field(None, description="Anti-position framing")
    split: Optional[str] = Field(None, description="Train/test split assignment")


class Statement(BaseModel):
    """A candidate position statement on an issue."""

    statement_id: str = Field(..., description="Unique statement identifier")
    issue_id: str = Field(..., description="Parent issue ID")
    text: str = Field(..., description="Full statement text")
    display_label: Optional[str] = Field(None, description="Display label (a, b, c, d)")
    provenance: Optional[str] = Field(None, description="Source: MODEL_MEDIATOR, HUMAN_CITIZEN, etc.")
    parent_statement_ids: Optional[List[str]] = Field(None, description="IDs of statements this was derived from")


class PreferenceRecord(BaseModel):
    """A human preference ranking/rating over statements."""

    preference_id: str = Field(..., description="Unique preference record ID")
    participant_id: str = Field(..., description="Anonymous participant identifier")
    issue_id: str = Field(..., description="Issue being evaluated")
    statement_ids: List[str] = Field(..., description="Statement IDs in display order")
    rankings: List[int] = Field(..., description="Numerical ranks (0=best)")
    agreements: Optional[List[str]] = Field(None, description="Agreement levels per statement")
    quality_ratings: Optional[List[str]] = Field(None, description="Quality ratings per statement")
    own_opinion_text: Optional[str] = Field(None, description="Participant's own written opinion")


class HabermasBank(BaseModel):
    """Complete Habermas Machine data bank."""

    version: str = Field("1.0.0", description="Schema version")
    created_at: datetime = Field(default_factory=datetime.now)
    issues: Dict[str, Issue] = Field(
        default_factory=dict, description="Dict of issue_id -> Issue"
    )
    statements: Dict[str, Statement] = Field(
        default_factory=dict, description="Dict of statement_id -> Statement"
    )
    preferences: List[PreferenceRecord] = Field(
        default_factory=list, description="All human preference records"
    )
