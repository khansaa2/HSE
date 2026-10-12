from __future__ import annotations

from pydantic import BaseModel, EmailStr, field_validator


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "VIEWER"

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"ADMIN", "OPERATOR", "VIEWER"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"Rôle invalide. Valeurs autorisées: {allowed}")
        return upper


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserRead(BaseModel):
    id: int
    username: str
    email: str
    role: str

    model_config = {"from_attributes": True}
