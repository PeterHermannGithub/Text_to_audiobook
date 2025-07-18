import re
import json

class PromptFactory:
    """
    Factory class responsible for constructing various prompts for the LLM.
    Separates prompt engineering logic from LLM orchestration.
    """

    def __init__(self):
        pass

    def create_structuring_prompt(self, text_content, context_hint="", text_metadata=None):
        """
        DEPRECATED: Old segmentation prompt. Use create_speaker_classification_prompt instead.
        Maintained for backward compatibility during transition.
        """
        return self.create_speaker_classification_prompt(text_content, text_metadata)
    
    def create_speaker_classification_prompt(self, numbered_lines, text_metadata=None):
        """
        SIMPLIFIED: Creates a much simpler speaker classification prompt.
        
        The LLM receives pre-segmented, numbered lines and must return exactly 
        the same number of speaker names in a JSON array.
        
        Args:
            numbered_lines: List of strings, each representing a pre-segmented line
            text_metadata: Metadata containing character names, format info, and context hints
            
        Returns:
            String prompt for speaker classification
        """
        if not numbered_lines:
            return ""
            
        # Extract POV information for narrative style rules
        pov_analysis = text_metadata.get('pov_analysis', {}) if text_metadata else {}
        pov_type = pov_analysis.get('type', 'UNKNOWN')
        narrator_id = pov_analysis.get('narrator_identifier', 'narrator')
        
        # Build dynamic POV rules
        pov_rules = self._build_dynamic_pov_rules(pov_type, narrator_id)
        
        # Extract character information
        character_profiles = text_metadata.get('character_profiles', []) if text_metadata else []
        character_list = self._build_character_list_for_pov(character_profiles, pov_type, narrator_id)
        
        # Build context block (if provided)
        context_hint = text_metadata.get('context_hint', {}) if text_metadata else {}
        context_block = ""
        if context_hint:
            rolling_context = self._build_rolling_context_section(context_hint)
            if rolling_context:
                context_block = f"\n\n---CONTEXT BLOCK (Previous text for context)---\n{rolling_context.strip()}"
        
        # Build task block
        task_block = ""
        for i, line in enumerate(numbered_lines, 1):
            task_block += f"{i}. {line}\n"
        
        task_block_line_count = len(numbered_lines)
        
        # ULTRATHINK SIMPLIFIED TEMPLATE - Critical instruction at the very end
        return f"""TASK: For each numbered line in the 'TASK BLOCK', identify the speaker.

NARRATIVE STYLE: {pov_rules}

CHARACTERS:
{character_list}{context_block}

---TASK BLOCK (Lines to classify)---
{task_block.strip()}

Your response MUST be a single, valid JSON array with exactly {task_block_line_count} string items."""

    def create_json_correction_prompt(self, malformed_json_text):
        """
        Creates a prompt to instruct the LLM to correct malformed JSON output.
        """
        return f"""🚨 JSON FIX REQUIRED

Your previous response was not valid JSON. You MUST fix it and respond with ONLY valid JSON.

❌ BROKEN JSON:
{malformed_json_text}

✅ REQUIRED FORMAT: 
["speaker1", "speaker2", "speaker3"]

🔧 COMMON FIXES NEEDED:
- Add missing quotes around strings
- Remove trailing commas
- Use double quotes, not single
- Ensure proper array format

⚠️ RESPOND WITH ONLY THE CORRECTED JSON ARRAY - NO OTHER TEXT:"""

    def create_character_description_prompt(self, character_name, dialogue_sample):
        """
        Creates a prompt to instruct the LLM to describe a character for voice casting.
        """
        return f"""Based on the following dialogue from '{character_name}', provide a concise, one-sentence description of their likely voice characteristics. 
        Focus on age, gender, and tone.

        Example: 'An elderly male with a deep, commanding voice.'
        Example: 'A young female with a bright, energetic voice.'

        Dialogue sample:
        ---
        {dialogue_sample}
        ---
        """

    def create_emotion_annotation_prompt(self, sentence):
        """
        Creates a prompt to instruct the LLM to annotate a sentence with emotions.
        """
        return f"""Analyze the following sentence and return a JSON object with three keys: "emotion", "pitch", and "speaking_rate".

        RULES:
        1.  "emotion" should be a single descriptive word (e.g., "happy", "sad", "angry", "calm").
        2.  "pitch" should be a float between -20.0 and 20.0.
        3.  "speaking_rate" should be a float between 0.25 and 4.0.

        EXAMPLE INPUT:
        "I'm so excited to go to the park!"

        EXAMPLE OUTPUT:
        {{
            "emotion": "excited",
            "pitch": 5.0,
            "speaking_rate": 1.2
        }}

        Now, process the following sentence:
        ---
        {sentence}
        ---
        """

    def _sanitize_text_for_llm(self, text):
        """
        Sanitizes text by replacing problematic Unicode characters and normalizing whitespace.
        """
        # Replace problematic Unicode quotes
        text = text.replace('「', '"').replace('」', '"')
        text = text.replace('『', '"').replace('』', '"')
        text = text.replace('《', '"').replace('》', '"')
        text = text.replace('〈', '"').replace('〉', '"')

        # Normalize all line endings to Unix-style
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Collapse multiple newlines into at most two (for paragraph breaks)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Collapse any remaining horizontal whitespace (spaces, tabs)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text
    
    def _infer_gender_from_pronouns(self, pronouns):
        """
        Infer gender hint from pronoun list for LLM context.
        
        Args:
            pronouns: List of pronouns associated with character
            
        Returns:
            String gender hint or None
        """
        pronoun_set = set(p.lower() for p in pronouns)
        
        # Check for male pronouns
        male_pronouns = {'he', 'him', 'his', 'himself'}
        if pronoun_set & male_pronouns:
            return 'male'
        
        # Check for female pronouns
        female_pronouns = {'she', 'her', 'hers', 'herself'}
        if pronoun_set & female_pronouns:
            return 'female'
        
        # Check for neutral/plural pronouns
        neutral_pronouns = {'they', 'them', 'their', 'theirs', 'themselves'}
        if pronoun_set & neutral_pronouns:
            return 'neutral'
        
        return None

    def _build_rolling_context_section(self, context_hint):
        """
        Build rolling context section for the prompt.
        
        Args:
            context_hint: Dictionary containing rolling context information
            
        Returns:
            String containing formatted rolling context section
        """
        if not context_hint:
            return ""
        
        context_sections = []
        
        # Recent speakers context
        if context_hint.get('recent_speakers'):
            recent_speakers = context_hint['recent_speakers']
            if recent_speakers:
                # Filter out generic speakers for better context
                character_speakers = [s for s in recent_speakers if s not in ['narrator', 'AMBIGUOUS']]
                if character_speakers:
                    context_sections.append(f"Recent speakers: {', '.join(character_speakers[-3:])}")
                else:
                    context_sections.append(f"Recent speakers: {', '.join(recent_speakers[-3:])}")
        
        # Conversation flow context
        if context_hint.get('conversation_flow'):
            flow_info = context_hint['conversation_flow']
            if isinstance(flow_info, list) and flow_info:
                flow_preview = []
                for flow_item in flow_info[-2:]:  # Last 2 conversation turns
                    speaker = flow_item.get('speaker', 'unknown')
                    text_preview = flow_item.get('text', '')[:40] + "..." if len(flow_item.get('text', '')) > 40 else flow_item.get('text', '')
                    flow_preview.append(f"{speaker}: {text_preview}")
                
                if flow_preview:
                    context_sections.append(f"Previous conversation:\n" + "\n".join(flow_preview))
        
        # Chunk position context (helps with continuity)
        if context_hint.get('chunk_position') is not None:
            chunk_pos = context_hint['chunk_position']
            if chunk_pos > 0:
                context_sections.append(f"This is continuation of text (chunk {chunk_pos + 1})")
        
        # Character introduction context
        if context_hint.get('introduced_characters'):
            introduced = context_hint['introduced_characters']
            if introduced:
                context_sections.append(f"Characters introduced in recent text: {', '.join(introduced[:5])}")
        
        # Build final context section
        if context_sections:
            context_text = "\n\n🔄 ROLLING CONTEXT (from previous text):\n" + "\n".join(f"• {section}" for section in context_sections)
            context_text += "\n(Use this context to maintain conversation flow and character consistency)"
            return context_text
        
        return ""

    def create_pov_aware_classification_prompt(self, task_lines, context_lines=None, text_metadata=None):
        """
        Creates a POV-aware speaker classification prompt using the Context vs Task model.
        
        This implements the Ultrathink architecture's Phase 2: Universal POV-Aware Prompting.
        Uses dynamic POV rules based on the narrative perspective analysis.
        
        Args:
            task_lines: List of lines to classify (the actual task)
            context_lines: List of context lines to provide background (not to classify)
            text_metadata: Metadata including POV analysis and character information
            
        Returns:
            String prompt optimized for the detected POV type
        """
        if not task_lines:
            return ""
        
        # Extract POV analysis from metadata
        pov_analysis = text_metadata.get('pov_analysis', {}) if text_metadata else {}
        pov_type = pov_analysis.get('type', 'UNKNOWN')
        narrator_id = pov_analysis.get('narrator_identifier', 'narrator')
        
        # Build dynamic POV rules based on analysis
        dynamic_pov_rules = self._build_dynamic_pov_rules(pov_type, narrator_id)
        
        # Extract character information
        character_profiles = text_metadata.get('character_profiles', []) if text_metadata else []
        character_list = self._build_character_list_for_pov(character_profiles, pov_type, narrator_id)
        
        # Build context block (if provided)
        context_block = ""
        if context_lines:
            context_block = self._build_context_block(context_lines)
        
        # Build task block
        task_block = self._build_task_block(task_lines)
        task_count = len(task_lines)
        
        # Build rolling context section if available
        rolling_context = ""
        context_hint = text_metadata.get('context_hint', {}) if text_metadata else {}
        if context_hint:
            rolling_context = self._build_rolling_context_section(context_hint)
        
        # Construct the universal POV-aware prompt
        return f"""TASK: Identify the speaker for each line in the 'LINES TO CLASSIFY' section.

NARRATIVE STYLE:
{dynamic_pov_rules}

SPEAKER TYPES:
- 'narrator': For descriptive text or thoughts in a third-person story.
- '{narrator_id}': For the main character's narration in a first-person story.
- Character names (e.g., 'Yoo Sangah'): For text inside quotation marks.
- 'AMBIGUOUS': If the speaker of dialogue is truly unclear.

KNOWN CHARACTERS:
{character_list}

--- CONTEXT BLOCK (DO NOT CLASSIFY THESE LINES) ---
{context_block}

--- LINES TO CLASSIFY (TASK BLOCK) ---
{task_block}

INSTRUCTIONS:
Respond with a valid JSON array containing exactly {task_count} speaker names. Your entire response must be only the JSON array.{rolling_context}"""

    def _build_dynamic_pov_rules(self, pov_type, narrator_id):
        """
        Build dynamic POV rules based on the detected narrative perspective.
        
        Args:
            pov_type: Detected POV type ('FIRST_PERSON', 'THIRD_PERSON', 'MIXED', etc.)
            narrator_id: Identifier for the narrator
            
        Returns:
            String containing POV-specific rules
        """
        if pov_type == 'FIRST_PERSON':
            return f"""This story is told in the first person. Lines from the 'I' or 'my' perspective are spoken by '{narrator_id}'.
Text describing actions, thoughts, or scenes from the narrator's perspective should be attributed to '{narrator_id}'.
Only dialogue within quotation marks spoken by other characters should be attributed to those character names."""
        
        elif pov_type == 'THIRD_PERSON':
            return f"""This story is told in the third person. Text describing scenes, actions, or thoughts is spoken by the 'narrator'.
Character dialogue within quotation marks should be attributed to the specific character speaking.
Avoid attributing narrative descriptions to characters unless they are clearly speaking."""
        
        elif pov_type == 'MIXED':
            return f"""This story uses mixed narrative perspectives. Text describing scenes or actions is usually spoken by the 'narrator'.
Lines from a first-person perspective ('I', 'my') may be spoken by '{narrator_id}' when clearly from the main character.
Character dialogue within quotation marks should be attributed to the specific character speaking."""
        
        else:  # UNKNOWN or other
            return f"""Analyze the narrative style carefully. Text describing scenes, actions, or thoughts is usually spoken by the 'narrator'.
Character dialogue within quotation marks should be attributed to the specific character speaking.
Lines from a clear first-person perspective may be spoken by a main character if identifiable."""

    def _build_character_list_for_pov(self, character_profiles, pov_type, narrator_id):
        """
        Build character list optimized for the detected POV type.
        
        Args:
            character_profiles: List of character profile dictionaries
            pov_type: Detected POV type
            narrator_id: Narrator identifier
            
        Returns:
            Formatted string of character information
        """
        if not character_profiles:
            if pov_type == 'FIRST_PERSON':
                return f"- {narrator_id} (main character/narrator)"
            else:
                return "No specific characters identified"
        
        character_lines = []
        
        # Add narrator if first person
        if pov_type == 'FIRST_PERSON':
            character_lines.append(f"- {narrator_id} (main character/narrator)")
        
        # Add other characters
        for profile in character_profiles[:10]:  # Limit to top 10
            name = profile['name']
            
            # Skip if this is the narrator in first person
            if pov_type == 'FIRST_PERSON' and name == narrator_id:
                continue
                
            profile_line = f"- {name}"
            
            # Add gender hints if available
            pronouns = profile.get('pronouns', [])
            if pronouns:
                gender_hint = self._infer_gender_from_pronouns(pronouns)
                if gender_hint:
                    profile_line += f" ({gender_hint})"
            
            # Add aliases if present
            aliases = profile.get('aliases', [])
            if aliases:
                profile_line += f" [also: {', '.join(aliases[:2])}]"
            
            character_lines.append(profile_line)
        
        return "\n".join(character_lines)

    def _build_context_block(self, context_lines):
        """
        Build the context block from provided context lines.
        
        Args:
            context_lines: List of context lines to display
            
        Returns:
            Formatted context block string
        """
        if not context_lines:
            return "(No context provided)"
        
        context_display = ""
        for i, line in enumerate(context_lines, 1):
            context_display += f"{i}. {line}\n"
        
        return context_display.strip()

    def _build_task_block(self, task_lines):
        """
        Build the task block from lines to classify.
        
        Args:
            task_lines: List of lines to classify
            
        Returns:
            Formatted task block string
        """
        task_display = ""
        for i, line in enumerate(task_lines, 1):
            task_display += f"{i}. {line}\n"
        
        return task_display.strip()
    
    def create_simple_classification_prompt(self, numbered_lines):
        """
        Creates a simplified speaker classification prompt for fallback scenarios.
        
        Args:
            numbered_lines: List of strings, each representing a pre-segmented line
            
        Returns:
            String prompt for simple speaker classification
        """
        if not numbered_lines:
            return ""
        
        # Build task block
        task_block = ""
        for i, line in enumerate(numbered_lines, 1):
            task_block += f"{i}. {line}\n"
        
        task_count = len(numbered_lines)
        
        return f"""Identify the speaker for each line. Respond with a JSON array of {task_count} speaker names.

Rules:
- Use "narrator" for descriptive text or scene descriptions
- Use character names for dialogue in quotes
- Use "AMBIGUOUS" if the speaker is unclear

Lines to classify:
{task_block.strip()}

Response format: ["speaker1", "speaker2", "speaker3"]"""

    def create_ultra_simple_prompt(self, numbered_lines):
        """
        Creates an ultra-simplified speaker classification prompt for final fallback.
        
        Args:
            numbered_lines: List of strings, each representing a pre-segmented line
            
        Returns:
            String prompt for ultra-simple speaker classification
        """
        if not numbered_lines:
            return ""
        
        # Build task block
        task_block = ""
        for i, line in enumerate(numbered_lines, 1):
            task_block += f"{i}. {line}\n"
        
        task_count = len(numbered_lines)
        
        return f"""For each line, say who is speaking. Return exactly {task_count} names in a JSON array.

{task_block.strip()}

Format: ["name1", "name2", "name3"]"""

    def create_batch_classification_prompt(self, batch_numbered_lines, text_metadata=None, context_hint=None):
        """
        Creates a batch processing prompt for multiple segments in a single request.
        
        This method creates a prompt that can handle multiple segments simultaneously,
        significantly reducing API call overhead for large documents.
        
        Args:
            batch_numbered_lines: List of lists, where each inner list contains lines for one segment
            text_metadata: Optional metadata for context
            context_hint: Optional context hint for processing
            
        Returns:
            String prompt for batch speaker classification
        """
        if not batch_numbered_lines:
            return ""
        
        # Calculate total lines and batch information
        total_lines = sum(len(lines) for lines in batch_numbered_lines)
        batch_count = len(batch_numbered_lines)
        batch_sizes = [len(lines) for lines in batch_numbered_lines]
        
        # Extract POV information for narrative style rules
        pov_analysis = text_metadata.get('pov_analysis', {}) if text_metadata else {}
        pov_type = pov_analysis.get('type', 'UNKNOWN')
        narrator_id = pov_analysis.get('narrator_identifier', 'narrator')
        
        # Build dynamic POV rules
        pov_rules = self._build_dynamic_pov_rules(pov_type, narrator_id)
        
        # Extract character information
        character_profiles = text_metadata.get('character_profiles', []) if text_metadata else []
        character_list = self._build_character_list_for_pov(character_profiles, pov_type, narrator_id)
        
        # Build context block
        context_block = ""
        if context_hint:
            rolling_context = self._build_rolling_context_section(context_hint)
            if rolling_context:
                context_block = f"\n\n---CONTEXT BLOCK (Previous text for context)---\n{rolling_context.strip()}"
        
        # Build batch task blocks
        batch_blocks = []
        for batch_idx, lines in enumerate(batch_numbered_lines):
            batch_block = f"BATCH {batch_idx + 1} ({len(lines)} lines):\n"
            for line_idx, line in enumerate(lines, 1):
                batch_block += f"{line_idx}. {line}\n"
            batch_blocks.append(batch_block.strip())
        
        # Create batch format explanation
        batch_format_example = []
        for batch_idx in range(batch_count):
            batch_size = batch_sizes[batch_idx]
            example_speakers = [f"speaker{i+1}" for i in range(min(batch_size, 3))]
            if batch_size > 3:
                example_speakers.append("...")
            batch_format_example.append(f"  [{', '.join(f'"{s}"' for s in example_speakers)}]")
        
        format_example = "[\n" + ",\n".join(batch_format_example) + "\n]"
        
        return f"""BATCH TASK: For each numbered line in each batch, identify the speaker.

NARRATIVE STYLE: {pov_rules}

CHARACTERS:
{character_list}{context_block}

---BATCH TASK BLOCKS (Process all {batch_count} batches)---
{chr(10).join(batch_blocks)}

Your response MUST be a single, valid JSON array containing {batch_count} sub-arrays, each with exactly the right number of speakers:

FORMAT EXAMPLE:
{format_example}

CRITICAL: Return exactly {batch_count} sub-arrays with sizes {batch_sizes} respectively."""

    def create_simple_batch_classification_prompt(self, batch_numbered_lines):
        """
        Creates a simplified batch processing prompt for fallback scenarios.
        
        Args:
            batch_numbered_lines: List of lists, where each inner list contains lines for one segment
            
        Returns:
            String prompt for simple batch speaker classification
        """
        if not batch_numbered_lines:
            return ""
        
        batch_count = len(batch_numbered_lines)
        batch_sizes = [len(lines) for lines in batch_numbered_lines]
        
        # Build simple batch blocks
        batch_blocks = []
        for batch_idx, lines in enumerate(batch_numbered_lines):
            batch_block = f"BATCH {batch_idx + 1}:\n"
            for line_idx, line in enumerate(lines, 1):
                batch_block += f"{line_idx}. {line}\n"
            batch_blocks.append(batch_block.strip())
        
        return f"""For each line in each batch, identify the speaker.

{chr(10).join(batch_blocks)}

Return exactly {batch_count} arrays with sizes {batch_sizes}.
Format: [["speaker1", "speaker2"], ["speaker3", "speaker4"]]"""
