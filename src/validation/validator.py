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
        
        # Enhanced quality report with detailed metrics
        ambiguous_count = sum(1 for seg in segments if seg.get('speaker') == 'AMBIGUOUS')
        unfixable_count = sum(1 for seg in segments if seg.get('speaker') == 'UNFIXABLE')
        attributed_count = sum(1 for seg in segments if seg.get('speaker') not in ['AMBIGUOUS', 'UNFIXABLE', None, ''])
        
        quality_report = {
            "quality_score": quality_score,
            "error_count": len(errors),  # Add backward compatibility field
            "quality_band": self._get_quality_band(quality_score),
            "attribution_metrics": {
                "total_segments": len(segments),
                "successfully_attributed": attributed_count,
                "ambiguous_segments": ambiguous_count,
                "unfixable_segments": unfixable_count,
                "attribution_success_rate": attributed_count / len(segments) if segments else 0.0
            },
            "error_analysis": {
                "total_errors": len(errors),
                "error_rate": len(errors) / len(segments) if segments else 0.0,
                "error_summary": self._summarize_errors(errors)
            },
            "detailed_errors": [e["message"] for e in errors]
        }
        
        self.logger.info(f"Validation completed: {quality_score:.1f}% quality ({self._get_quality_band(quality_score)}), "
                         f"{len(errors)} errors, {attributed_count}/{len(segments)} attributed, "
                         f"{ambiguous_count} ambiguous, {unfixable_count} unfixable")
        
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
        """
        Calculate overall quality score using HARSH but realistic scoring.
        
        RECALIBRATED Quality Bands (Harsh but Truthful):
        - Excellent (95-100%): Near-perfect attribution with <5% error rate
        - Good (80-94%): High-quality attribution with <15% error rate  
        - Fair (60-79%): Acceptable attribution with <30% error rate
        - Poor (30-59%): Significant issues with 30-60% error rate
        - Critical (0-29%): Severe problems with >60% error rate
        
        NOTE: This scoring is intentionally harsh to provide truthful quality assessment.
        A 36% error rate should score ~40% (Poor), not 98% (Excellent).
        """
        
        if not segments:
            return 0.0
        
        total_segments = len(segments)
        error_count = len(errors)
        error_rate = error_count / total_segments if total_segments > 0 else 1.0
        
        # Calculate base attribution success rate
        successfully_attributed = sum(1 for seg in segments 
                                    if seg.get('speaker') not in ['AMBIGUOUS', 'UNFIXABLE', None, ''])
        attribution_success_rate = successfully_attributed / total_segments
        
        # HARSH SCORING: Error rate directly impacts quality
        # If error rate is 36%, quality should be around 40% (64% penalty)
        error_penalty_factor = min(1.0, error_rate * 2.0)  # Double penalty for errors
        
        # Start with attribution success, but apply harsh error penalty
        base_score = attribution_success_rate * 100.0 * (1.0 - error_penalty_factor)
        
        # Additional penalties for specific quality issues
        ambiguous_count = sum(1 for seg in segments if seg.get('speaker') == 'AMBIGUOUS')
        unfixable_count = sum(1 for seg in segments if seg.get('speaker') == 'UNFIXABLE')
        
        ambiguous_ratio = ambiguous_count / total_segments
        unfixable_ratio = unfixable_count / total_segments
        
        # HARSH ambiguity penalties - every AMBIGUOUS segment hurts quality significantly
        ambiguity_penalty = ambiguous_ratio * 30.0  # 30 points per 100% ambiguous
        
        # SEVERE unfixable penalty - these are complete failures
        unfixable_penalty = unfixable_ratio * 50.0  # 50 points per 100% unfixable
        
        # Apply additional error type penalties
        error_type_penalty = self._calculate_harsh_error_penalties(errors, total_segments)
        
        # Calculate final score with all penalties
        final_score = base_score - ambiguity_penalty - unfixable_penalty - error_type_penalty
        
        # Ensure score stays within bounds
        final_score = max(0.0, min(100.0, final_score))
        
        # Log detailed scoring breakdown for transparency
        quality_band = self._get_quality_band(final_score)
        self.logger.info(f"HARSH Quality Scoring: error_rate={error_rate:.2%}, "
                        f"base_score={base_score:.1f}, ambiguity_penalty={ambiguity_penalty:.1f}, "
                        f"unfixable_penalty={unfixable_penalty:.1f}, error_penalty={error_type_penalty:.1f}, "
                        f"final={final_score:.1f}% ({quality_band})")
        
        return final_score
    
    def _calculate_harsh_error_penalties(self, errors: List[Dict[str, Any]], total_segments: int) -> float:
        """
        Calculate HARSH error penalties that directly reflect quality issues.
        Returns penalty points to subtract from quality score.
        """
        if total_segments == 0 or not errors:
            return 0.0
        
        # HARSH error penalties that reflect true quality impact
        error_penalties = {
            'missing_speaker': 8.0,      # Critical - completely unusable segment
            'missing_text': 10.0,        # Critical - no content  
            'empty_text': 5.0,           # High - wasted segment
            'dialogue_as_narrator': 3.0, # Moderate - speaker misattribution
            'narrative_as_dialogue': 3.0,# Moderate - speaker misattribution
            'unknown_speaker': 2.0,      # Moderate - attribution confusion
            'very_long_dialogue': 1.0,   # Minor - segmentation issue
            'consecutive_same_speaker': 0.5, # Minor - possible oversegmentation
            'too_many_single_use_speakers': 5.0  # High - systematic attribution failure
        }
        
        # Calculate total penalty
        total_penalty = 0.0
        error_counts = {}
        
        for error in errors:
            error_type = error.get('type', 'unknown')
            penalty = error_penalties.get(error_type, 1.0)  # Default penalty for unknown errors
            total_penalty += penalty
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Scale penalty based on error density
        error_ratio = len(errors) / total_segments
        
        # Additional scaling for high error density (when errors exceed 50% of segments)
        if error_ratio > 0.5:
            # Compound penalty for systematic failure
            density_multiplier = 1.0 + (error_ratio - 0.5)  # 1.0 to 1.5x
            total_penalty *= density_multiplier
        
        # Cap maximum penalty to prevent negative scores in extreme cases
        max_penalty = 60.0  # Maximum 60 points penalty from error types
        return min(total_penalty, max_penalty)
    
    def _calculate_error_impact(self, errors: List[Dict[str, Any]], total_segments: int) -> float:
        """
        LEGACY: Calculate error impact score with progressive penalties.
        Returns score out of 30 points.
        
        NOTE: This method is kept for compatibility but is replaced by 
        _calculate_harsh_error_penalties in the new scoring system.
        """
        if total_segments == 0:
            return 0.0
        
        # Recalibrated error penalties (much more lenient)
        error_penalties = {
            'missing_speaker': 2.0,      # Reduced from 10
            'missing_text': 3.0,         # Reduced from 10
            'empty_text': 1.0,           # Reduced from 5
            'dialogue_as_narrator': 0.5, # Reduced from 3
            'narrative_as_dialogue': 0.5,# Reduced from 3
            'unknown_speaker': 0.3,      # Reduced from 2
            'very_long_dialogue': 0.1,   # Reduced from 1
            'consecutive_same_speaker': 0.1, # Reduced from 1
            'too_many_single_use_speakers': 1.0  # Reduced from 5
        }
        
        # Calculate total penalty
        total_penalty = 0.0
        error_counts = {}
        
        for error in errors:
            error_type = error.get('type', 'unknown')
            penalty = error_penalties.get(error_type, 0.2)  # Reduced default
            total_penalty += penalty
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Progressive penalty scaling - diminishing returns for many errors
        error_ratio = len(errors) / total_segments
        if error_ratio > 0.5:
            # Cap penalty impact for very error-heavy content
            total_penalty = min(total_penalty, 15.0)
        
        # Return error impact score (30 points max, minus penalties)
        return max(0.0, 30.0 - total_penalty)
    
    def _calculate_content_preservation(self, segments: List[Dict[str, Any]]) -> float:
        """
        Calculate content preservation score.
        Returns score out of 10 points.
        """
        if not segments:
            return 0.0
        
        # Check for content integrity indicators
        total_chars = sum(len(seg.get('text', '')) for seg in segments)
        non_empty_segments = sum(1 for seg in segments if seg.get('text', '').strip())
        
        # Basic content preservation checks
        preservation_score = 10.0
        
        # Penalty for too many empty segments
        empty_ratio = (len(segments) - non_empty_segments) / len(segments)
        if empty_ratio > 0.1:  # More than 10% empty
            preservation_score -= empty_ratio * 20
        
        # Penalty for suspiciously short total content
        if total_chars < 100:  # Very short content
            preservation_score -= 5.0
        
        return max(0.0, preservation_score)
    
    def _get_quality_band(self, score: float) -> str:
        """
        Get quality band description for a score using HARSH but realistic thresholds.
        
        RECALIBRATED Quality Bands (Harsh but Truthful):
        - Excellent (95-100%): Near-perfect attribution with <5% error rate
        - Good (80-94%): High-quality attribution with <15% error rate  
        - Fair (60-79%): Acceptable attribution with <30% error rate
        - Poor (30-59%): Significant issues with 30-60% error rate
        - Critical (0-29%): Severe problems with >60% error rate
        """
        if score >= 95:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 30:
            return "Poor"
        else:
            return "Critical"
    
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
    
    def _summarize_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Summarize errors by type for the quality report.
        """
        error_summary = {}
        for error in errors:
            error_type = error.get('type', 'unknown')
            error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        # Sort by frequency
        return dict(sorted(error_summary.items(), key=lambda x: x[1], reverse=True))