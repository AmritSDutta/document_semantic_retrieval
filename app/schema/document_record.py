from typing import List

from pydantic import BaseModel, constr, Field, conint, ConfigDict


class SearchRequest(BaseModel):
    search_term: constr(max_length=1000) = Field(
        ...,
        json_schema_extra={"example": "foo"},
        description="Search term (max 100 characters)"
    )
    limit: conint(le=5) = Field(
        3,
        json_schema_extra={"example": 3},
        description="Number of results to return (max 5)"
    )


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume_id: str
    name: str
    education: str
    category: str
    skills: list[str]
    summary: str


class Topic(BaseModel):
    name: str
    confidence: float


class ClassificationResult(BaseModel):
    result: List[Topic]

    @property
    def sorted_result(self) -> List[Topic]:
        return sorted(self.result, key=lambda t: t.confidence, reverse=True)

    @property
    def derive_relevant_topic(self, n: int = 5) -> str:
        sorted_topic = sorted(self.result, key=lambda t: t.confidence, reverse=True)
        op_n_topics = sorted_topic[:n]

        # 3. Extract just the names and join them with a comma
        top_names = [topic.name for topic in op_n_topics]
        return ", ".join(top_names)


class PassageRequest(BaseModel):
    passage: str = 'python is silly at sometime'
