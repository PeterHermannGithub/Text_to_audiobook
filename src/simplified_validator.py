import logging
from typing import Dict, List, Tuple, Any

class SimplifiedValidator:
    """
    Simplified validator for the new architecture that focuses on speaker attribution quality.
    
    Since the new architecture never modifies text content (deterministic segmentation + 
    classification-only LLM), we no longer need fuzzy text matching for content preservation.
    This validator focuses purely on speaker attribution quality and consistency.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate(self, processed_data: List[Tuple[Dict[str, Any], int]], original_text: str, text_metadata: Dict[str, Any]) -> Tuple[List[Tuple[Dict[str, Any], int]], Dict[str, Any]]:
        """
        Validates speaker attribution quality and generates a quality report.
        
        Args:
            processed_data: List of (segment_dict, chunk_index) tuples
            original_text: Original text (used for length validation only)
            text_metadata: Metadata with character profiles and names
            
        Returns:
            Tuple of (validated_data, quality_report)
        """
        self.logger.info(f"Starting simplified validation on {len(processed_data)} segments")
        
        errors = []
        known_character_names = text_metadata.get('potential_character_names', set())
        character_profiles = text_metadata.get('character_profiles', [])
        
        # Extract segments for analysis
        segments = [item[0] for item in processed_data]
        
        # Validation checks
        self._validate_data_integrity(segments, errors)
        self._validate_speaker_quality(segments, known_character_names, character_profiles, errors)
        self._validate_dialogue_consistency(segments, errors)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(segments, errors, len(original_text))
        
        # Add error flags to segments for downstream processing
        self._add_error_flags_to_segments(processed_data, errors)
        
        quality_report = {
            "quality_score": quality_score,
            "error_count": len(errors),
            "ambiguous_count": sum(1 for seg in segments if seg.get('speaker') == 'AMBIGUOUS'),
            "total_segments": len(segments),
            "errors": [e["message"] for e in errors]
        }
        
        self.logger.info(f"Validation completed: {quality_score:.1f}% quality, {len(errors)} errors, {quality_report['ambiguous_count']} ambiguous")
        
        return processed_data, quality_report
    
    def _validate_data_integrity(self, segments: List[Dict[str, Any]], errors: List[Dict[str, Any]]):
        """Validate basic data integrity (no null values, empty segments, etc.)."""
        for i, segment in enumerate(segments):
            # Check for missing speaker
            if not segment.get('speaker'):
                errors.append({
                    "index": i,
                    "type": "missing_speaker", 
                    "message": f"Segment {i} has missing or null speaker"
                })
                continue
                
            # Check for missing text
            if not segment.get('text'):
                errors.append({
                    "index": i,
                    "type": "missing_text",
                    "message": f"Segment {i} has missing or null text"
                })
                continue
                
            # Check for empty text
            if not segment['text'].strip():
                errors.append({
                    "index": i,
                    "type": "empty_text",
                    "message": f"Segment {i} has empty text content"
                })
    
    def _validate_speaker_quality(self, segments: List[Dict[str, Any]], known_names: set, character_profiles: List[Dict[str, Any]], errors: List[Dict[str, Any]]):
        """Validate speaker attribution quality and consistency."""
        
        # Build comprehensive character name list
        all_character_names = set(known_names)
        for profile in character_profiles:
            all_character_names.add(profile['name'])
            all_character_names.update(profile.get('aliases', []))
        
        # Track speaker usage patterns
        speaker_counts = {}
        dialogue_speakers = set()
        
        for i, segment in enumerate(segments):
            speaker = (segment.get('speaker') or '').strip()
            text = (segment.get('text') or '').strip()
            
            if not speaker:
                continue
                
            # Count speaker usage
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            # Track dialogue speakers
            if self._is_dialogue_text(text):
                dialogue_speakers.add(speaker)
                
                # Check for dialogue without proper attribution
                if speaker == 'narrator':
                    errors.append({
                        "index": i,
                        "type": "dialogue_as_narrator", 
                        "message": f"Segment {i} appears to be dialogue but attributed to narrator"
                    })
                
                # Check for unknown speakers in dialogue
                if speaker not in all_character_names and speaker not in ['AMBIGUOUS', 'narrator']:
                    # Check if it's a reasonable character name
                    if not self._is_reasonable_character_name(speaker):
                        errors.append({
                            "index": i,
                            "type": "unknown_speaker",
                            "message": f"Segment {i} has unknown speaker '{speaker}' not in character list"
                        })
            
            # Check for narrative attributed to characters
            elif not self._is_dialogue_text(text) and speaker not in ['narrator', 'AMBIGUOUS']:
                # This might be narrative incorrectly attributed to a character
                if not self._could_be_character_thought(text):
                    errors.append({
                        "index": i,
                        "type": "narrative_as_dialogue",
                        "message": f"Segment {i} appears to be narrative but attributed to character '{speaker}'"
                    })
        
        # Check for speakers used only once (might be errors)
        single_use_speakers = {speaker for speaker, count in speaker_counts.items() 
                              if count == 1 and speaker not in ['narrator', 'AMBIGUOUS']}
        
        if len(single_use_speakers) > len(all_character_names) * 0.5:  # More than 50% are single-use
            errors.append({
                "index": -1,
                "type": "too_many_single_use_speakers",
                "message": f"High number of single-use speakers ({len(single_use_speakers)}) suggests attribution errors"
            })
    
    def _validate_dialogue_consistency(self, segments: List[Dict[str, Any]], errors: List[Dict[str, Any]]):
        """Validate dialogue attribution consistency and conversation flow."""
        
        # Track conversation patterns
        recent_speakers = []
        
        for i, segment in enumerate(segments):
            speaker = (segment.get('speaker') or '').strip()
            text = (segment.get('text') or '').strip()
            
            if not self._is_dialogue_text(text):
                continue
                
            # Track recent dialogue speakers
            if speaker not in ['narrator', 'AMBIGUOUS']:
                recent_speakers.append(speaker)
                if len(recent_speakers) > 10:  # Keep only last 10 speakers
                    recent_speakers.pop(0)
            
            # Check for very long speeches (might be segmentation errors)
            if len(text) > 500:  # Very long dialogue segment
                errors.append({
                    "index": i,
                    "type": "very_long_dialogue",
                    "message": f"Segment {i} has very long dialogue ({len(text)} chars) - possible segmentation error"
                })
            
            # Check for back-to-back identical speakers in dialogue
            if i > 0:
                prev_segment = segments[i-1]
                prev_speaker = prev_segment.get('speaker', '').strip()
                prev_text = prev_segment.get('text', '').strip()
                
                if (speaker == prev_speaker and 
                    self._is_dialogue_text(prev_text) and
                    speaker not in ['narrator', 'AMBIGUOUS']):
                    # Same character speaking consecutively - might be correct but worth noting
                    if len(text) < 100 and len(prev_text) < 100:  # Both are short
                        errors.append({
                            "index": i,
                            "type": "consecutive_same_speaker",
                            "message": f"Segments {i-1} and {i} both attributed to '{speaker}' - might need merging"
                        })
    
    def _calculate_quality_score(self, segments: List[Dict[str, Any]], errors: List[Dict[str, Any]], original_length: int) -> float:
        """Calculate overall quality score based on various factors."""
        
        if not segments:
            return 0.0
            
        # Base score starts at 100
        score = 100.0
        
        # Deduct points for errors
        error_penalties = {
            'missing_speaker': 10,
            'missing_text': 10, 
            'empty_text': 5,
            'dialogue_as_narrator': 3,
            'narrative_as_dialogue': 3,
            'unknown_speaker': 2,
            'very_long_dialogue': 1,
            'consecutive_same_speaker': 1,
            'too_many_single_use_speakers': 5
        }
        
        for error in errors:
            penalty = error_penalties.get(error['type'], 2)  # Default penalty
            score -= penalty
        
        # Deduct points for high ambiguity
        ambiguous_count = sum(1 for seg in segments if seg.get('speaker') == 'AMBIGUOUS')
        ambiguous_ratio = ambiguous_count / len(segments)
        
        if ambiguous_ratio > 0.3:  # More than 30% ambiguous
            score -= (ambiguous_ratio - 0.3) * 100  # Heavy penalty for high ambiguity
        elif ambiguous_ratio > 0.1:  # More than 10% ambiguous  
            score -= (ambiguous_ratio - 0.1) * 50   # Moderate penalty
        
        # Ensure score doesn't go below 0
        return max(0.0, score)
    
    def _add_error_flags_to_segments(self, processed_data: List[Tuple[Dict[str, Any], int]], errors: List[Dict[str, Any]]):
        """Add error flags to segments for downstream processing."""
        for error in errors:
            segment_index = error.get("index", -1)
            if 0 <= segment_index < len(processed_data):
                segment_dict = processed_data[segment_index][0]
                if 'errors' not in segment_dict:
                    segment_dict['errors'] = []
                segment_dict['errors'].append(error["type"])
    
    def _is_dialogue_text(self, text: str) -> bool:
        """Check if text appears to be dialogue."""
        dialogue_markers = ['"', '"', '"', "'", '—', '–', ':']
        return any(marker in text for marker in dialogue_markers)
    
    def _is_reasonable_character_name(self, name: str) -> bool:
        """Check if a name looks like a reasonable character name."""
        if not name or len(name) < 2 or len(name) > 50:
            return False
            
        # Must start with capital letter
        if not name[0].isupper():
            return False
            
        # Check for common non-name words
        non_names = {
            'the', 'and', 'but', 'for', 'with', 'from', 'into', 'then',
            'here', 'there', 'this', 'that', 'what', 'where', 'when'
        }
        
        return name.lower() not in non_names
    
    def _could_be_character_thought(self, text: str) -> bool:
        """Check if narrative text could be internal character thoughts."""
        thought_indicators = ['thought', 'wondered', 'realized', 'remembered', 'considered']
        return any(indicator in text.lower() for indicator in thought_indicators)