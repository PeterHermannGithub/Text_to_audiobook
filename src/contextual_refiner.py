import logging
from typing import List, Dict, Any, Optional
from collections import deque

class ContextualRefiner:
    """
    Enhanced refiner that uses contextual memory and conversation flow 
    to resolve AMBIGUOUS speaker attributions intelligently.
    
    This refiner focuses on turn-taking patterns, conversation flow,
    and proximity-based context to make educated speaker predictions.
    """
    
    def __init__(self, llm_orchestrator):
        self.llm_orchestrator = llm_orchestrator
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.context_window = 5  # Number of previous segments to consider
        self.max_iterations = 2  # Maximum refinement iterations
        
    def refine_ambiguous_speakers(self, structured_segments: List[Dict[str, Any]], text_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Refine segments with AMBIGUOUS or problematic speaker attributions using contextual memory.
        
        Args:
            structured_segments: List of segments with speaker/text
            text_metadata: Rich metadata including character profiles
            
        Returns:
            List of segments with improved speaker attributions
        """
        self.logger.info(f"Starting contextual refinement on {len(structured_segments)} segments")
        
        # Find all segments that need refinement (AMBIGUOUS + dialogue misattributed as Unfixable)
        refinement_indices = []
        for i, segment in enumerate(structured_segments):
            speaker = segment.get('speaker', '').strip()
            text = segment.get('text', '').strip()
            errors = segment.get('errors', [])
            
            # Standard AMBIGUOUS segments
            if speaker == 'AMBIGUOUS':
                refinement_indices.append(i)
            # Unfixable segments that contain dialogue (likely misclassified)
            elif speaker.lower() in ['unfixable', 'UNFIXABLE'] and self._is_dialogue_text(text):
                self.logger.debug(f"Found Unfixable dialogue segment {i}: {repr(text[:50])}")
                refinement_indices.append(i)
            # Segments with dialogue_as_narrator errors (narrator assigned to dialogue)
            elif 'dialogue_as_narrator' in errors and self._is_dialogue_text(text):
                self.logger.debug(f"Found dialogue_as_narrator error segment {i}: {repr(text[:50])}")
                refinement_indices.append(i)
        
        if not refinement_indices:
            self.logger.info("No AMBIGUOUS or problematic segments found, skipping refinement")
            return structured_segments
        
        ambiguous_count = sum(1 for i in refinement_indices if structured_segments[i].get('speaker') == 'AMBIGUOUS')
        unfixable_count = len(refinement_indices) - ambiguous_count
        self.logger.info(f"Found {ambiguous_count} AMBIGUOUS and {unfixable_count} problematic segments to refine")
        
        # ENHANCEMENT: Pre-process segments for duplicate detection and merging
        refined_segments = self._preprocess_segments_for_duplicates(structured_segments.copy())
        
        # Update refinement indices after potential segment merging
        refinement_indices = self._update_refinement_indices_after_preprocessing(refined_segments)
        
        # Process each problematic segment
        successful_refinements = 0
        
        for segment_idx in refinement_indices:
            old_speaker = refined_segments[segment_idx]['speaker']
            self.logger.debug(f"Refining segment {segment_idx} ({old_speaker}): {repr(refined_segments[segment_idx]['text'][:100])}")
            
            # Build conversation context
            context = self._build_conversation_context(refined_segments, segment_idx)
            
            # Generate contextual refinement prompt
            refined_speaker = self._get_contextual_speaker_prediction(
                context, refined_segments[segment_idx]['text'], text_metadata
            )
            
            if refined_speaker and refined_speaker not in ['AMBIGUOUS', 'UNFIXABLE']:
                # Clean up any error flags since we successfully resolved them
                if 'errors' in refined_segments[segment_idx]:
                    refined_segments[segment_idx]['errors'] = []
                
                refined_segments[segment_idx]['speaker'] = refined_speaker
                refined_segments[segment_idx]['refined'] = True
                refined_segments[segment_idx]['refinement_method'] = 'contextual_memory'
                
                self.logger.debug(f"Refined segment {segment_idx}: {old_speaker} -> {refined_speaker}")
                successful_refinements += 1
            else:
                # If we couldn't refine but it was Unfixable dialogue, convert to AMBIGUOUS
                if old_speaker.lower() in ['unfixable'] and self._is_dialogue_text(refined_segments[segment_idx]['text']):
                    refined_segments[segment_idx]['speaker'] = 'AMBIGUOUS'
                    refined_segments[segment_idx]['refinement_method'] = 'unfixable_to_ambiguous'
                    self.logger.debug(f"Converted segment {segment_idx}: {old_speaker} -> AMBIGUOUS")
                else:
                    self.logger.debug(f"Could not refine segment {segment_idx}, keeping {old_speaker}")
        
        self.logger.info(f"Successfully refined {successful_refinements}/{len(refinement_indices)} problematic segments")
        return refined_segments
    
    def _build_conversation_context(self, segments: List[Dict[str, Any]], target_index: int) -> Dict[str, Any]:
        """
        Build conversation context for a target segment.
        
        Args:
            segments: All segments
            target_index: Index of the segment to refine
            
        Returns:
            Context dictionary with conversation flow information
        """
        context = {
            'previous_speakers': [],
            'conversation_flow': [],
            'recent_dialogue': [],
            'target_text': segments[target_index]['text']
        }
        
        # Collect context from previous segments
        start_idx = max(0, target_index - self.context_window)
        
        for i in range(start_idx, target_index):
            segment = segments[i]
            speaker = segment.get('speaker', 'unknown')
            text = segment.get('text', '')
            
            # Track speaker sequence
            if speaker not in ['narrator', 'AMBIGUOUS', 'unknown']:
                context['previous_speakers'].append(speaker)
            
            # Track conversation flow
            context['conversation_flow'].append({
                'index': i,
                'speaker': speaker,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'is_dialogue': self._is_dialogue_text(text)
            })
            
            # Collect recent dialogue for pattern analysis
            if self._is_dialogue_text(text):
                context['recent_dialogue'].append({
                    'speaker': speaker,
                    'text': text
                })
        
        return context
    
    def _get_contextual_speaker_prediction(self, context: Dict[str, Any], target_text: str, text_metadata: Dict[str, Any]) -> Optional[str]:
        """
        ENHANCED: Get speaker prediction using contextual memory and conversation flow with narrator protection.
        
        Args:
            context: Conversation context
            target_text: Text of the ambiguous segment
            text_metadata: Rich metadata with character profiles
            
        Returns:
            Predicted speaker name or None
        """
        # ENHANCEMENT: Narrator protection for non-dialogue text
        narrator_protection = self._apply_narrator_protection(target_text, context)
        if narrator_protection:
            self.logger.debug(f"Narrator protection applied: {narrator_protection}")
            return narrator_protection
        
        # First, try turn-taking pattern analysis
        turn_taking_prediction = self._analyze_turn_taking_patterns(context, target_text)
        if turn_taking_prediction:
            self.logger.debug(f"Turn-taking prediction: {turn_taking_prediction}")
            return turn_taking_prediction
        
        # If turn-taking fails, use LLM with rich context
        return self._get_llm_contextual_prediction(context, target_text, text_metadata)
    
    def _apply_narrator_protection(self, text: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Apply narrator protection to prevent internal monologue from being misattributed.
        
        Args:
            text: Text to check
            context: Conversation context
            
        Returns:
            'narrator' if protection should be applied, None otherwise
        """
        # Get dialogue confidence
        dialogue_confidence = self._get_dialogue_confidence(text)
        
        # Strong protection: Very low dialogue confidence should remain narrator
        if dialogue_confidence < 0.3:
            self.logger.debug(f"Strong narrator protection: dialogue confidence {dialogue_confidence:.2f}")
            return 'narrator'
        
        # Medium protection: Text without quotation marks but with internal thought patterns
        if dialogue_confidence < 0.6:
            import re
            
            # Check for internal thought indicators
            internal_indicators = [
                r'\b(?:he|she|they|it)\s+(?:thought|wondered|considered|realized|remembered|knew|felt|understood)',
                r'\b(?:his|her|their|its)\s+(?:thoughts?|mind|memory|feelings?)',
                r'\bwas\s+(?:thinking|wondering|considering)',
                r'\bit\s+(?:wasn\'t|was|seemed|appeared)',
                r'\b(?:the\s+)?(?:genre|novel|story|book)\s+(?:of|was|had)',
            ]
            
            for pattern in internal_indicators:
                if re.search(pattern, text, re.IGNORECASE):
                    self.logger.debug(f"Medium narrator protection: internal thought pattern detected")
                    return 'narrator'
        
        # No protection needed
        return None
    
    def _analyze_turn_taking_patterns(self, context: Dict[str, Any], target_text: str) -> Optional[str]:
        """
        ENHANCED: Analyze conversation turn-taking patterns to predict speaker with content awareness.
        
        Args:
            context: Conversation context
            target_text: Text to attribute
            
        Returns:
            Predicted speaker based on turn-taking patterns
        """
        # ENHANCEMENT 1: Check for explicit attributions first (highest priority)
        explicit_speaker = self._detect_explicit_attribution(target_text)
        if explicit_speaker:
            self.logger.debug(f"Found explicit attribution: {explicit_speaker}")
            return explicit_speaker
        
        # ENHANCEMENT 2: Content-based duplicate detection
        duplicate_speaker = self._check_for_content_duplicates(context, target_text)
        if duplicate_speaker:
            self.logger.debug(f"Found content duplicate, using previous attribution: {duplicate_speaker}")
            return duplicate_speaker
        
        # ENHANCEMENT 3: Stronger dialogue requirement for character attribution
        if not self._is_strong_dialogue_text(target_text):
            return 'narrator'  # Non-dialogue should remain narrator
        
        previous_speakers = context['previous_speakers']
        if not previous_speakers:
            return None  # No context to work with
        
        # ENHANCEMENT 4: Content-aware turn-taking patterns
        return self._apply_content_aware_turn_taking(previous_speakers, target_text, context)
    
    def _detect_explicit_attribution(self, text: str) -> Optional[str]:
        """
        Detect explicit speaker attributions like '"Hello," said John.'
        
        Args:
            text: Text to check for explicit attribution
            
        Returns:
            Speaker name if explicitly attributed, None otherwise
        """
        import re
        
        # Common attribution patterns
        attribution_patterns = [
            r'"[^"]*"\s*,?\s*(\w+(?:\s+\w+)*)\s+(?:said|asked|replied|answered|whispered|shouted|exclaimed|muttered|cried)',
            r'(\w+(?:\s+\w+)*)\s+(?:said|asked|replied|answered|whispered|shouted|exclaimed|muttered|cried)\s*,?\s*"[^"]*"',
            r'"[^"]*"\s*—\s*(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s*:\s*"[^"]*"',  # Script format
        ]
        
        for pattern in attribution_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                speaker_name = match.group(1).strip()
                # Basic validation - should be a proper name
                if speaker_name and speaker_name[0].isupper() and len(speaker_name) > 1:
                    # Filter out common non-speaker words
                    non_speakers = {'said', 'asked', 'replied', 'answered', 'whispered', 'shouted', 'exclaimed', 'muttered', 'cried', 'he', 'she', 'they', 'it'}
                    if speaker_name.lower() not in non_speakers:
                        return speaker_name.title()
        
        return None
    
    def _check_for_content_duplicates(self, context: Dict[str, Any], target_text: str) -> Optional[str]:
        """
        Check if the target text appears in recent conversation flow.
        If so, use the same speaker attribution.
        
        Args:
            context: Conversation context
            target_text: Text to check for duplicates
            
        Returns:
            Speaker from previous identical content, None if no duplicates
        """
        target_clean = target_text.strip().lower()
        
        # Check recent conversation flow for identical content
        for flow_item in reversed(context.get('conversation_flow', [])):
            flow_text = flow_item.get('text', '').strip().lower()
            flow_speaker = flow_item.get('speaker', '')
            
            # If we find identical text with a non-ambiguous speaker
            if (flow_text == target_clean and 
                flow_speaker not in ['AMBIGUOUS', 'UNFIXABLE', 'unknown', 'narrator']):
                return flow_speaker
        
        return None
    
    def _is_strong_dialogue_text(self, text: str) -> bool:
        """
        Enhanced dialogue detection requiring stronger evidence.
        
        Args:
            text: Text to check
            
        Returns:
            True if text shows strong evidence of being dialogue
        """
        # Must have quotation marks for strong dialogue classification
        has_quotes = any(marker in text for marker in ['"', '"', '"'])
        
        # Or other strong dialogue indicators
        has_dialogue_markers = any(marker in text for marker in ['—', '–'])
        has_script_format = ':' in text and text.strip().endswith('"')
        
        return has_quotes or has_dialogue_markers or has_script_format
    
    def _apply_content_aware_turn_taking(self, previous_speakers: List[str], target_text: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Apply turn-taking patterns with content awareness.
        
        Args:
            previous_speakers: List of previous speakers
            target_text: Text to attribute
            context: Full conversation context
            
        Returns:
            Predicted speaker based on enhanced turn-taking logic
        """
        # Pattern 1: Simple alternation (A -> B -> A -> B) - but with content validation
        if len(previous_speakers) >= 2:
            last_speaker = previous_speakers[-1]
            second_last_speaker = previous_speakers[-2]
            
            # If the last two speakers were different, expect alternation
            if last_speaker != second_last_speaker:
                predicted_speaker = second_last_speaker
                
                # VALIDATION: Check if this attribution makes sense given content
                if self._validate_speaker_content_match(predicted_speaker, target_text, context):
                    return predicted_speaker
        
        # Pattern 2: Avoid immediate repetition with content awareness
        if len(previous_speakers) >= 1:
            last_speaker = previous_speakers[-1]
            unique_speakers = list(set(previous_speakers))
            
            # If there are multiple speakers, avoid immediate repetition
            if len(unique_speakers) > 1:
                other_speakers = [s for s in unique_speakers if s != last_speaker]
                if other_speakers:
                    # Find the most contextually appropriate speaker
                    for speaker in reversed(previous_speakers):
                        if (speaker != last_speaker and 
                            self._validate_speaker_content_match(speaker, target_text, context)):
                            return speaker
        
        return None  # No clear pattern
    
    def _validate_speaker_content_match(self, speaker: str, text: str, context: Dict[str, Any]) -> bool:
        """
        Validate if a speaker attribution makes sense given the content.
        
        Args:
            speaker: Proposed speaker
            text: Text content
            context: Conversation context
            
        Returns:
            True if the attribution seems reasonable
        """
        # Always accept narrator for non-dialogue
        if speaker == 'narrator' and not self._is_strong_dialogue_text(text):
            return True
        
        # For character speakers, require dialogue-like text
        if speaker != 'narrator' and not self._is_strong_dialogue_text(text):
            return False
        
        # Check if this speaker has spoken recently (recency bias)
        recent_speakers = context.get('previous_speakers', [])[-3:]  # Last 3 speakers
        if speaker in recent_speakers:
            return True
        
        # If speaker hasn't spoken recently, require stronger evidence
        # (This prevents random speaker assignments)
        return len(recent_speakers) < 2  # Only allow if we don't have much context
        
        return True
    
    def _get_llm_contextual_prediction(self, context: Dict[str, Any], target_text: str, text_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Use LLM with rich conversational context to predict speaker.
        
        Args:
            context: Conversation context
            target_text: Text to attribute
            text_metadata: Rich metadata with character profiles
            
        Returns:
            LLM-predicted speaker name
        """
        try:
            # Build contextual prompt
            prompt = self._build_contextual_prompt(context, target_text, text_metadata)
            
            # Get LLM response
            response_text = self.llm_orchestrator._get_llm_response(prompt)
            
            # Parse and validate response
            predicted_speaker = self._parse_speaker_prediction(response_text, text_metadata)
            
            return predicted_speaker
            
        except Exception as e:
            self.logger.warning(f"LLM contextual prediction failed: {e}")
            return None
    
    def _build_contextual_prompt(self, context: Dict[str, Any], target_text: str, text_metadata: Dict[str, Any]) -> str:
        """
        Build a contextual prompt for speaker prediction.
        
        Args:
            context: Conversation context
            target_text: Text to attribute
            text_metadata: Rich metadata
            
        Returns:
            Contextual prompt string
        """
        # Extract character information
        character_profiles = text_metadata.get('character_profiles', [])
        character_names = [profile['name'] for profile in character_profiles]
        
        # Build character context
        character_context = ""
        if character_profiles:
            context_lines = []
            for profile in character_profiles[:8]:  # Limit to 8 characters
                line = f"- {profile['name']}"
                if profile.get('pronouns'):
                    gender = self._infer_gender_from_pronouns(profile['pronouns'])
                    if gender:
                        line += f" ({gender})"
                if profile.get('titles'):
                    line += f" [titles: {', '.join(profile['titles'][:2])}]"
                context_lines.append(line)
            character_context = "\\n".join(context_lines)
        else:
            character_context = ", ".join(character_names) if character_names else "None identified"
        
        # Build conversation flow
        flow_context = ""
        if context['conversation_flow']:
            flow_lines = []
            for i, step in enumerate(context['conversation_flow'][-4:], 1):  # Last 4 turns
                speaker = step['speaker'] if step['speaker'] != 'AMBIGUOUS' else '?'
                text_preview = step['text'][:60] + "..." if len(step['text']) > 60 else step['text']
                flow_lines.append(f"{i}. {speaker}: {text_preview}")
            flow_context = "\\n".join(flow_lines)
        
        prompt = f"""You are analyzing conversation flow to identify a speaker. You must distinguish between spoken dialogue, internal thoughts, and narrative description.

CHARACTERS:
{character_context}

RECENT CONVERSATION:
{flow_context}

TARGET LINE TO ATTRIBUTE:
"{target_text}"

CRITICAL INSTRUCTIONS:
1. DIALOGUE (Character Attribution): Only if the text is SPOKEN OUT LOUD
   - Text should have quotation marks ("") or clear speech indicators
   - Examples: "Hello there," she said. OR "How are you?"
   
2. INTERNAL THOUGHTS (Narrator Attribution): Character's private thoughts in third-person narrative
   - Text like: "He thought about..." OR "She realized that..." OR "It wasn't the most popular..."
   - NO quotation marks, describes mental state/thoughts
   
3. NARRATIVE DESCRIPTION (Narrator Attribution): General story narration
   - Describes actions, settings, or events
   - Examples: "There was a pause." OR "Someone coughed."

4. Consider conversation flow ONLY for actual spoken dialogue
5. Respond with character name ONLY for spoken dialogue, "narrator" for everything else
6. If genuinely uncertain between characters for dialogue, respond with "AMBIGUOUS"

ANALYSIS:
- Does this text have quotation marks or clear speech indicators? 
- Is this describing thoughts, feelings, or internal state?
- Is this general narrative description?

SPEAKER:"""
        
        return prompt
    
    def _parse_speaker_prediction(self, response_text: str, text_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Parse and validate LLM speaker prediction response.
        
        Args:
            response_text: Raw LLM response
            text_metadata: Metadata for validation
            
        Returns:
            Validated speaker name or None
        """
        if not response_text:
            return None
            
        # Clean response
        predicted_speaker = response_text.strip().strip('"\'').strip()
        
        # Validate against known characters
        character_names = {profile['name'] for profile in text_metadata.get('character_profiles', [])}
        character_names.update(text_metadata.get('potential_character_names', set()))
        
        # Accept exact matches
        if predicted_speaker in character_names:
            return predicted_speaker
        
        # Accept special speakers
        if predicted_speaker.lower() in ['narrator', 'ambiguous']:
            return predicted_speaker.lower()
        
        # Try fuzzy matching for character names
        if character_names:
            from fuzzywuzzy import process
            best_match = process.extractOne(predicted_speaker, list(character_names))
            if best_match and best_match[1] > 80:  # 80% similarity threshold
                return best_match[0]
        
        # If no good match, return None
        self.logger.debug(f"Could not validate speaker prediction: {predicted_speaker}")
        return None
    
    def _is_dialogue_text(self, text: str) -> bool:
        """
        ENHANCED: Check if text appears to be dialogue with improved accuracy.
        
        This method is more conservative to prevent internal monologue from being
        misclassified as dialogue.
        """
        return self._get_dialogue_confidence(text) > 0.5
    
    def _get_dialogue_confidence(self, text: str) -> float:
        """
        Get confidence score (0.0-1.0) that text is dialogue.
        
        Args:
            text: Text to analyze
            
        Returns:
            Confidence score where 1.0 = definitely dialogue, 0.0 = definitely not dialogue
        """
        if not text or not text.strip():
            return 0.0
        
        confidence = 0.0
        text_clean = text.strip()
        
        # Strong positive indicators (high confidence)
        if any(marker in text for marker in ['"', '"', '"']):
            confidence += 0.8  # Quotation marks are strong dialogue indicators
            
        # Medium positive indicators
        if any(marker in text for marker in ['—', '–']):
            confidence += 0.4  # Em/en dashes often indicate dialogue
            
        if text_clean.endswith(':') or text_clean.startswith('"'):
            confidence += 0.3  # Script format or opening quote
            
        # Speech verb patterns
        import re
        speech_verbs = r'\b(?:said|asked|replied|answered|whispered|shouted|exclaimed|muttered|cried|thought|wondered)\b'
        if re.search(speech_verbs, text, re.IGNORECASE):
            confidence += 0.2
        
        # Negative indicators (reduce confidence)
        # Internal thought patterns
        internal_patterns = [
            r'\b(?:he|she|they|it)\s+(?:thought|wondered|considered|realized|remembered|knew|felt|understood)',
            r'\b(?:his|her|their|its)\s+(?:thoughts?|mind|memory|feelings?)',
            r'\bwas\s+(?:thinking|wondering|considering)',
            r'\bit\s+(?:wasn\'t|was|seemed|appeared)',
        ]
        
        for pattern in internal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                confidence -= 0.3
                
        # Narrative description patterns
        narrative_patterns = [
            r'\b(?:the|a|an)\s+\w+\s+(?:was|were|had|did|could|would|should)',
            r'\b(?:there|here)\s+(?:was|were|had)',
            r'\b(?:it|this|that)\s+(?:was|were|had|seemed|appeared)',
        ]
        
        for pattern in narrative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                confidence -= 0.2
        
        # Ensure confidence stays in valid range
        return max(0.0, min(1.0, confidence))
    
    def _preprocess_segments_for_duplicates(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess segments to detect and merge consecutive duplicates.
        
        This addresses the issue where identical text gets split into multiple segments
        and then attributed to different speakers by the turn-taking logic.
        
        Args:
            segments: List of segments to preprocess
            
        Returns:
            List of segments with duplicates merged
        """
        if not segments:
            return segments
        
        processed_segments = []
        current_segment = None
        merge_count = 0
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', '')
            
            # Skip empty segments
            if not text:
                continue
            
            # If this is the first segment or differs from current
            if current_segment is None:
                current_segment = segment.copy()
                processed_segments.append(current_segment)
            else:
                current_text = current_segment.get('text', '').strip()
                
                # Check for exact duplicate text
                if text == current_text:
                    # Found duplicate - merge with current segment
                    merge_count += 1
                    self.logger.debug(f"Merging duplicate segment {i}: {repr(text[:50])}")
                    
                    # Preserve metadata from both segments
                    self._merge_segment_metadata(current_segment, segment)
                    
                    # Continue to next segment without adding this one
                    continue
                else:
                    # Different text - start new current segment
                    current_segment = segment.copy()
                    processed_segments.append(current_segment)
        
        if merge_count > 0:
            self.logger.info(f"Merged {merge_count} duplicate segments ({len(segments)} -> {len(processed_segments)})")
        
        return processed_segments
    
    def _merge_segment_metadata(self, target_segment: Dict[str, Any], source_segment: Dict[str, Any]):
        """
        Merge metadata from source segment into target segment.
        
        Args:
            target_segment: Segment to merge into (modified in place)
            source_segment: Segment to merge from
        """
        # Combine error lists
        target_errors = target_segment.get('errors', [])
        source_errors = source_segment.get('errors', [])
        if source_errors:
            all_errors = list(set(target_errors + source_errors))
            target_segment['errors'] = all_errors
        
        # Track merge in metadata
        if 'merged_count' not in target_segment:
            target_segment['merged_count'] = 1
        target_segment['merged_count'] += 1
        
        # Preserve refinement information if present
        if source_segment.get('refined'):
            target_segment['merged_refined'] = True
        
        # Update confidence if present
        source_confidence = source_segment.get('confidence', 0)
        target_confidence = target_segment.get('confidence', 0)
        if source_confidence > 0 or target_confidence > 0:
            target_segment['confidence'] = max(source_confidence, target_confidence)
    
    def _update_refinement_indices_after_preprocessing(self, preprocessed_segments: List[Dict[str, Any]]) -> List[int]:
        """
        Update refinement indices after segment preprocessing/merging.
        
        Args:
            preprocessed_segments: Segments after duplicate merging
            
        Returns:
            Updated list of indices that need refinement
        """
        refinement_indices = []
        
        for i, segment in enumerate(preprocessed_segments):
            speaker = segment.get('speaker', '').strip()
            text = segment.get('text', '').strip()
            errors = segment.get('errors', [])
            
            # Standard AMBIGUOUS segments
            if speaker == 'AMBIGUOUS':
                refinement_indices.append(i)
            # Unfixable segments that contain dialogue (likely misclassified)
            elif speaker.lower() in ['unfixable', 'UNFIXABLE'] and self._is_dialogue_text(text):
                refinement_indices.append(i)
            # Segments with dialogue_as_narrator errors
            elif 'dialogue_as_narrator' in errors and self._is_dialogue_text(text):
                refinement_indices.append(i)
        
        return refinement_indices

    def _infer_gender_from_pronouns(self, pronouns: List[str]) -> Optional[str]:
        """Infer gender from pronoun list."""
        pronoun_set = set(p.lower() for p in pronouns)
        
        male_pronouns = {'he', 'him', 'his', 'himself'}
        if pronoun_set & male_pronouns:
            return 'male'
        
        female_pronouns = {'she', 'her', 'hers', 'herself'}
        if pronoun_set & female_pronouns:
            return 'female'
        
        neutral_pronouns = {'they', 'them', 'their', 'theirs', 'themselves'}
        if pronoun_set & neutral_pronouns:
            return 'neutral'
        
        return None