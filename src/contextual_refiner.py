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
        
        # Process each problematic segment
        refined_segments = structured_segments.copy()
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
        Get speaker prediction using contextual memory and conversation flow.
        
        Args:
            context: Conversation context
            target_text: Text of the ambiguous segment
            text_metadata: Rich metadata with character profiles
            
        Returns:
            Predicted speaker name or None
        """
        # First, try turn-taking pattern analysis
        turn_taking_prediction = self._analyze_turn_taking_patterns(context, target_text)
        if turn_taking_prediction:
            self.logger.debug(f"Turn-taking prediction: {turn_taking_prediction}")
            return turn_taking_prediction
        
        # If turn-taking fails, use LLM with rich context
        return self._get_llm_contextual_prediction(context, target_text, text_metadata)
    
    def _analyze_turn_taking_patterns(self, context: Dict[str, Any], target_text: str) -> Optional[str]:
        """
        Analyze conversation turn-taking patterns to predict speaker.
        
        Args:
            context: Conversation context
            target_text: Text to attribute
            
        Returns:
            Predicted speaker based on turn-taking patterns
        """
        if not self._is_dialogue_text(target_text):
            return 'narrator'  # Non-dialogue is usually narrator
        
        previous_speakers = context['previous_speakers']
        if not previous_speakers:
            return None  # No context to work with
        
        # Pattern 1: Simple alternation (A -> B -> A -> B)
        if len(previous_speakers) >= 2:
            last_speaker = previous_speakers[-1]
            second_last_speaker = previous_speakers[-2]
            
            # If the last two speakers were different, expect alternation
            if last_speaker != second_last_speaker:
                return second_last_speaker  # Return to previous speaker
        
        # Pattern 2: Avoid immediate repetition
        if len(previous_speakers) >= 1:
            last_speaker = previous_speakers[-1]
            unique_speakers = list(set(previous_speakers))
            
            # If there are multiple speakers, avoid immediate repetition
            if len(unique_speakers) > 1:
                other_speakers = [s for s in unique_speakers if s != last_speaker]
                if other_speakers:
                    # Prefer the most recent other speaker
                    for speaker in reversed(previous_speakers):
                        if speaker != last_speaker:
                            return speaker
        
        return None  # No clear pattern
    
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
        
        prompt = f"""You are analyzing conversation flow to identify a speaker. Based on the conversation context and turn-taking patterns, identify who is most likely speaking the target line.

CHARACTERS:
{character_context}

RECENT CONVERSATION:
{flow_context}

TARGET LINE TO ATTRIBUTE:
"{target_text}"

INSTRUCTIONS:
1. Consider conversation flow and turn-taking patterns
2. Use character context and previous attributions
3. Respond with ONLY the character name (or "narrator" for narrative text)
4. If truly uncertain, respond with "AMBIGUOUS"

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
        """Check if text appears to be dialogue."""
        dialogue_markers = ['"', '"', '"', "'", '—', '–']
        return any(marker in text for marker in dialogue_markers)
    
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