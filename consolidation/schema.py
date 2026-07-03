from dataclasses import dataclass, asdict, field
from uuid import uuid4
from typing import Dict, Any, Union, List


@dataclass
class Message:
    """
    Represents a message in a conversation.

    Attributes:
        message_id (str): Unique identifier for the message
        role (str): Role of the message sender
        content (str): Content of the message
    """

    message_id: str
    role: str
    content: str

    def to_dict(self):
        """
        Convert the Message object to a dictionary.

        Returns:
            dict: Dictionary representation of the Message
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """
        Create a Message object from a dictionary.

        Args:
            data (dict): Dictionary containing message data

        Returns:
            Message: New Message object
        """

        return cls(
            message_id=data.get("message_id", uuid4().hex),
            role=data.get("role", ""),
            content=data.get("content", ""),
        )


@dataclass
class Session:
    """
    Represents a conversation session with metadata.

    Attributes:
        session_id (str): Unique identifier for the session
        session_date (str): Date and time of the session in format 'YYYY-MM-DD Weekday HH:MM:SS'
        conversation (list): List of Message objects in the conversation
        others (dict): Additional metadata for the session
    """

    session_id: str
    session_date: str
    conversation: list[Message]
    others: dict

    """def __post_init__(self):
        
        #Validates the session_date format after initialization.
        #Prints a warning if the session_date doesn't match the expected format.
        
        try:
            # Simple validation using existing function
            str_to_datetime(self.session_date)
        except ValueError:
            print(f"Warning: Invalid session_date format: '{self.session_date}'.")"""

    def to_dict(self):
        """
        Convert the Session object to a dictionary.

        Returns:
            dict: Dictionary representation of the Session
        """
        return {
            "session_id": self.session_id,
            "session_date": self.session_date,
            "conversation": [
                msg.to_dict() if isinstance(msg, Message) else msg
                for msg in self.conversation
            ],
            "others": self.others,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a Session object from a dictionary.

        Args:
            data (dict): Dictionary containing session data

        Returns:
            Session: New Session object
        """
        # Convert conversation messages
        conversation = []
        for msg_data in data.get("conversation", []):
            if isinstance(msg_data, dict):
                conversation.append(Message.from_dict(msg_data))
            else:
                conversation.append(msg_data)

        return cls(
            session_id=data.get("session_id", uuid4().hex),
            session_date=data.get("session_date", ""),
            conversation=conversation,
            others=data.get("others", {}),
        )


@dataclass
class QA:
    """
    Represents a question-answer pair with metadata.

    Attributes:
        question_id (str): Unique identifier for the question
        question_type (str): Type or category of question
        question_date (str): Date when the question was asked
        question (str): The question text
        session_pool (list): List of session ids
        answer (str): The answer text
        abs_or_adversarial (bool): Whether the question is an abstention or adversarial
        evidence (dict): Evidence supporting the answer, with structure:
        {
            'sessions': {session_id: Session object or dict},
            'messages': {message_id: Message object or dict}
        }
    """

    question_id: str
    question_type: str
    question_date: str
    question: str
    session_pool: list[str]
    answer: str
    abs_or_adversarial: bool = False
    evidence: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {"sessions": [], "messages": []}
    )
    others: dict = field(default_factory=dict)

    def to_dict(self):
        """
        Convert the QA object to a dictionary.

        Returns:
            dict: Dictionary representation of the QA
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """
        Create a QA object from a dictionary.

        Args:
            data (dict): Dictionary containing QA data

        Returns:
            QA: New QA object
        """
        evidence = {"sessions": {}, "messages": {}}

        # Process evidence if provided
        if "evidence" in data and isinstance(data["evidence"], dict):
            # Process session evidence
            for session_id, session_data in data["evidence"].get("sessions", {}).items():
                if isinstance(session_data, dict):
                    evidence["sessions"][session_id] = Session.from_dict(session_data)
                else:
                    evidence["sessions"][session_id] = session_data

            # Process message evidence
            for message_id, message_data in data["evidence"].get("messages", {}).items():
                if isinstance(message_data, dict):
                    evidence["messages"][message_id] = Message.from_dict(message_data)
                else:
                    evidence["messages"][message_id] = message_data

        return cls(
            question_id=data.get("question_id", uuid4().hex),
            question_type=data.get("question_type", ""),
            question_date=data.get("question_date", ""),
            question=data.get("question", ""),
            session_pool=data.get("session_pool", []),
            answer=data.get("answer", ""),
            evidence=evidence,
        )

    def add_session_evidence(self, session_id: str, session: Union[Session, dict]):
        """
        Add a session as evidence.

        Args:
            session_id (str): ID of the session
            session (Session or dict): Session object or dictionary
        """
        self.evidence["sessions"][session_id] = session

    def add_message_evidence(self, message_id: str, message: Union[Message, dict]):
        """
        Add a message as evidence.

        Args:
            message_id (str): ID of the message
            message (Message or dict): Message object or dictionary
        """
        self.evidence["messages"][message_id] = message


@dataclass
class Result:
    question_id: str
    question: str
    question_type: str
    answer: str
    retrieved_results: List[Dict]
    # retrieved_results_key: List[Any]
    response: Dict[str, Any]
    context: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class DeperecatedResult:
    question_id: str
    question: str
    answer: str
    retrieved_results: List[Dict]
    response: Dict[str, Any]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
