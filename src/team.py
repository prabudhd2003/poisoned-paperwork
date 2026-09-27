"""Fixed team-to-worker mapping used by every CARC GPU stage."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamMember:
    user_id: str
    name: str
    worker_id: int


TEAM = {
    "user1": TeamMember("user1", "Prabudhd", 0),
    "user2": TeamMember("user2", "Gary", 1),
    "user3": TeamMember("user3", "Saaketh", 2),
    "user4": TeamMember("user4", "Khalid", 3),
    "user5": TeamMember("user5", "Shail", 4),
}


def get_team_member(user_id: str) -> TeamMember:
    """Return the fixed member record or raise a useful error."""
    try:
        return TEAM[user_id]
    except KeyError as exc:
        choices = ", ".join(TEAM)
        raise ValueError(f"Unknown user ID {user_id!r}; choose one of: {choices}") from exc
