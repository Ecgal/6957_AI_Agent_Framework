"""
MiniCPM message serializer for browser_use integration.
Handles conversion of browser_use messages to MiniCPM format with image support.
Optimized for low memory usage.
"""

import base64
import io
from typing import Any, Optional

from PIL import Image

from Agents.browser_use.llm.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    UserMessage,
)


class MiniCPMMessageSerializer:
    """Serializer for converting messages to MiniCPM format with memory optimization."""

    @staticmethod
    def _extract_text_content(content: Any, max_length: Optional[int] = None) -> str:
        """
        Extract text content from message content, ignoring images.
        
        Args:
            content: Message content
            max_length: Optional maximum text length (truncate if exceeded)
        """
        if content is None:
            return ''
        if isinstance(content, str):
            text = content
        else:
            text_parts: list[str] = []
            for part in content:
                if hasattr(part, 'type'):
                    if part.type == 'text':
                        text_parts.append(part.text)
                    elif part.type == 'refusal':
                        text_parts.append(f'[Refusal] {part.refusal}')
                elif isinstance(part, dict):
                    if part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                    elif part.get('type') == 'refusal':
                        text_parts.append(f"[Refusal] {part.get('refusal', '')}")
                # Skip image parts as they're handled separately

            text = ' '.join(text_parts) if text_parts else ''
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length] + '\n...[truncated for memory]'
        
        return text

    @staticmethod
    def _extract_image(content: Any) -> Optional[Image.Image]:
        """Extract first image from message content."""
        if content is None or isinstance(content, str):
            return None

        for part in content:
            # Handle Pydantic ContentPartImageParam
            if hasattr(part, 'type') and part.type == 'image_url':
                url = part.image_url.url if hasattr(part.image_url, 'url') else str(part.image_url)
                return MiniCPMMessageSerializer._load_image_from_url(url)
            # Handle dict format
            elif isinstance(part, dict) and part.get('type') == 'image_url':
                image_url = part.get('image_url', {})
                url = image_url.get('url', '') if isinstance(image_url, dict) else str(image_url)
                if url:
                    return MiniCPMMessageSerializer._load_image_from_url(url)
            # Handle direct PIL Image
            elif isinstance(part, Image.Image):
                return part

        return None

    @staticmethod
    def _load_image_from_url(url: str, max_size: int = 2880) -> Optional[Image.Image]:
        """
        Load PIL Image from URL (base64 data URI or file path) and aggressively resize.
        
        Args:
            url: Image URL (base64 data URI or file path)
            max_size: Maximum width/height. Default 240 for memory efficiency.
        
        Returns:
            PIL Image in RGB format, resized to fit within max_size x max_size
        """
        try:
            if url.startswith('data:image'):
                # Base64 encoded image: data:image/jpeg;base64,<data>
                _, data = url.split(',', 1)
                image_bytes = base64.b64decode(data)
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                # File path
                import os
                if os.path.exists(url):
                    img = Image.open(url).convert('RGB')
                else:
                    return None
            
            # ALWAYS resize to max_size for memory efficiency
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return img
        except Exception:
            return None

    @staticmethod
    def serialize_messages(
        messages: list[BaseMessage],
        max_text_length: int = 20000,  # Limit text to 20k chars
        image_max_size: int = 2880,      # Smaller images for memory
    ) -> tuple[list[dict[str, Any]], Optional[Image.Image], Optional[str]]:
        """
        Convert browser_use messages to MiniCPM format with aggressive memory optimization.

        MiniCPM requires:
        1. System message prepended to first user message (no separate system role)
        2. First user message with format: {"role": "user", "content": [image, text]}
        3. Other messages with format: {"role": "user/assistant", "content": text}

        Args:
            messages: List of BaseMessage objects from browser_use
            max_text_length: Maximum length for each text content (default 20000)
            image_max_size: Maximum image dimension (default 240)

        Returns:
            Tuple of (minicpm_messages, first_image, system_message) where:
            - minicpm_messages: List of dicts in MiniCPM chat format
            - first_image: PIL Image from first user message (or None)
            - system_message: Combined system message text (for logging)
        """
        # Deep copy to avoid modifying original messages
        messages = [m.model_copy(deep=True) for m in messages]

        minicpm_messages: list[dict[str, Any]] = []
        system_parts: list[str] = []
        first_image: Optional[Image.Image] = None
        first_user_processed = False

        for message in messages:
            role = message.role if hasattr(message, 'role') else None

            # Extract system messages
            if isinstance(message, SystemMessage) or role in ['system', 'developer']:
                text = MiniCPMMessageSerializer._extract_text_content(
                    message.content,
                    max_length=max_text_length
                )
                if text:
                    system_parts.append(text)
                continue

            # Determine role for non-system messages
            if isinstance(message, UserMessage):
                role = 'user'
            elif isinstance(message, AssistantMessage):
                role = 'assistant'
            else:
                role = 'user'

            # Extract text and image
            text_content = MiniCPMMessageSerializer._extract_text_content(
                message.content,
                max_length=max_text_length
            )
            
            # Only extract image from first user message
            if role == 'user' and not first_user_processed:
                message_image = MiniCPMMessageSerializer._extract_image(message.content)
                if message_image:
                    # Resize image
                    if message_image.width > image_max_size or message_image.height > image_max_size:
                        message_image.thumbnail((image_max_size, image_max_size), Image.Resampling.LANCZOS)
                    first_image = message_image
                first_user_processed = True

            # Build message dict
            if not minicpm_messages and first_image and role == 'user':
                # First message with image
                minicpm_messages.append({
                    "role": "user",
                    "content": [first_image, text_content]
                })
            else:
                # Text-only message
                minicpm_messages.append({
                    "role": role,
                    "content": text_content
                })

        # Prepend system message to first user message (with truncation)
        system_message = None
        if system_parts:
            system_message = '\n\n'.join(system_parts)
            # Truncate combined system message if too long
            if len(system_message) > max_text_length:
                system_message = system_message[:max_text_length] + '\n...[truncated for memory]'
            
            if minicpm_messages and minicpm_messages[0]['role'] == 'user':
                first_msg = minicpm_messages[0]
                if isinstance(first_msg['content'], list):
                    # Has image: [image, text]
                    combined_text = f"{system_message}\n\n{first_msg['content'][1]}"
                    # Ensure combined text doesn't exceed limit
                    if len(combined_text) > max_text_length:
                        combined_text = combined_text[:max_text_length] + '\n...[truncated for memory]'
                    first_msg['content'][1] = combined_text
                else:
                    # Text only
                    combined_text = f"{system_message}\n\n{first_msg['content']}"
                    if len(combined_text) > max_text_length:
                        combined_text = combined_text[:max_text_length] + '\n...[truncated for memory]'
                    first_msg['content'] = combined_text
        print(f"\n{'=' * 100}")
        print(f"[SERIALIZER DEBUG]")
        print(f"{'=' * 100}")
        print(f"Total messages to serialize: {len(messages)}")
        print(f"First image: {first_image.size if first_image else 'None'}")
        print(f"System message length: {len(system_message) if system_message else 0}")

        if minicpm_messages:
            first_msg = minicpm_messages[0]
            if isinstance(first_msg['content'], list):
                text_part = first_msg['content'][1]
            else:
                text_part = first_msg['content']

            # Find browser_state section
            if '<browser_state>' in text_part:
                start = text_part.find('<browser_state>')
                end = text_part.find('</browser_state>') + len('</browser_state>')
                browser_state_section = text_part[start:end]

                print(f"\nBrowser state preview (first 1500 chars):")
                print(browser_state_section[:1500])

                # Count indexes
                indexes = re.findall(r'\[(\d+)\]', browser_state_section)
                print(f"\nFound {len(indexes)} indexes in browser_state")
                if indexes:
                    print(f"First 20 indexes: {indexes[:20]}")
            else:
                print("\nWARNING: No <browser_state> found in message!")
                print(f"Message preview (first 500 chars):")
                print(text_part[:500])

        print(f"{'=' * 100}\n")

        return minicpm_messages, first_image, system_message
