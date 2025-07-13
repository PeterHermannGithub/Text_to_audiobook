import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from fuzzywuzzy import fuzz, process

class UnfixableRecoverySystem:
    """
    Progressive fallback system for recovering UNFIXABLE segments.
    
    Implements multiple strategies to avoid marking segments as UNFIXABLE:
    1. Heuristic speaker detection using context patterns
    2. Conservative speaker assignment based on content analysis
    3. Content-type classification for narrator vs dialogue
    4. Proximity-based assignment using nearby segments
    5. Statistical pattern matching from successful attributions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Recovery strategy configuration
        self.context_window = 3  # Number of surrounding segments to consider
        self.min_confidence_threshold = 0.4  # Minimum confidence for recovery
        self.statistical_sample_size = 10  # Minimum segments for pattern analysis
        
        # Dialogue indicators for content-type classification
        self.dialogue_indicators = ['"', '"', '"', "'", '—', '–', ':']
        self.narrative_indicators = [
            'he ', 'she ', 'they ', 'it ', 'the ', 'his ', 'her ', 'their ',
            'was ', 'were ', 'had ', 'would ', 'could ', 'should '
        ]
        
        # Action/thought patterns that often indicate narrative
        self.narrative_patterns = [
            r'\b(?:thought|wondered|realized|remembered|considered|decided|knew|felt|believed)\b',
            r'\b(?:looked|turned|walked|stepped|moved|gestured|nodded|smiled|frowned)\b',
            r'\b(?:was|were|had|would|could|should)\s+\w+ing\b',
            r'\bthe\s+\w+\s+(?:was|were|had|would|could|should)\b'
        ]
        
        # Speaker transition indicators
        self.speaker_transition_patterns = [
            r'(?:said|asked|replied|whispered|shouted|muttered|cried|exclaimed)',
            r'(?:continued|added|answered|responded|declared|announced)',
            r'(?:turned to|looked at|faced|addressed)'
        ]
    
    def recover_unfixable_segments(self, structured_segments: List[Dict[str, Any]], 
                                  text_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main entry point for UNFIXABLE segment recovery.
        
        Args:
            structured_segments: List of segments, some potentially marked as UNFIXABLE
            text_metadata: Rich metadata including character profiles
            
        Returns:
            List of segments with UNFIXABLE segments recovered where possible
        """
        self.logger.info(f"Starting UNFIXABLE recovery on {len(structured_segments)} segments")
        
        # Find UNFIXABLE segments
        unfixable_indices = []
        for i, segment in enumerate(structured_segments):
            if segment.get('speaker') == 'UNFIXABLE':
                unfixable_indices.append(i)
        
        if not unfixable_indices:
            self.logger.info("No UNFIXABLE segments found")
            return structured_segments
        
        self.logger.info(f"Found {len(unfixable_indices)} UNFIXABLE segments to recover")
        
        # Build statistical patterns from successfully attributed segments
        attribution_patterns = self._build_attribution_patterns(structured_segments, text_metadata)
        
        # Process each UNFIXABLE segment
        recovered_segments = structured_segments.copy()
        recovery_stats = {
            'attempted': 0,
            'recovered': 0,
            'strategies_used': defaultdict(int)
        }
        
        for segment_idx in unfixable_indices:
            recovery_stats['attempted'] += 1
            
            self.logger.debug(f"Attempting recovery for segment {segment_idx}")
            
            # Try progressive recovery strategies
            recovered_speaker = self._attempt_progressive_recovery(
                recovered_segments, segment_idx, text_metadata, attribution_patterns
            )
            
            if recovered_speaker and recovered_speaker != 'UNFIXABLE':
                old_speaker = recovered_segments[segment_idx]['speaker']
                recovered_segments[segment_idx]['speaker'] = recovered_speaker
                recovered_segments[segment_idx]['recovered'] = True
                recovered_segments[segment_idx]['recovery_method'] = recovered_speaker.get('method', 'unknown')
                
                # Track recovery method
                method = recovered_speaker.get('method', 'unknown') if isinstance(recovered_speaker, dict) else 'heuristic'
                recovery_stats['strategies_used'][method] += 1
                recovery_stats['recovered'] += 1
                
                self.logger.debug(f"Recovered segment {segment_idx}: {old_speaker} -> {recovered_speaker}")
            else:
                self.logger.debug(f"Could not recover segment {segment_idx}")
        
        success_rate = (recovery_stats['recovered'] / recovery_stats['attempted'] * 100) if recovery_stats['attempted'] > 0 else 0
        self.logger.info(f"Recovery completed: {recovery_stats['recovered']}/{recovery_stats['attempted']} "
                         f"({success_rate:.1f}%) segments recovered")
        
        if recovery_stats['strategies_used']:
            strategy_summary = ', '.join(f"{method}: {count}" for method, count in recovery_stats['strategies_used'].items())
            self.logger.info(f"Recovery methods used: {strategy_summary}")
        
        return recovered_segments
    
    def _attempt_progressive_recovery(self, segments: List[Dict[str, Any]], target_idx: int,
                                    text_metadata: Dict[str, Any], 
                                    attribution_patterns: Dict[str, Any]) -> Optional[str]:
        """
        Attempt progressive recovery strategies for a single UNFIXABLE segment.
        
        Returns the recovered speaker name or None if recovery fails.
        """
        target_segment = segments[target_idx]
        text = target_segment.get('text', '')
        
        # Strategy 1: Heuristic speaker detection
        speaker = self._heuristic_speaker_detection(text, text_metadata)
        if speaker:
            self.logger.debug(f"Strategy 1 (heuristic) succeeded: {speaker}")
            return speaker
        
        # Strategy 2: Context-based recovery
        speaker = self._context_based_recovery(segments, target_idx, text_metadata)
        if speaker:
            self.logger.debug(f"Strategy 2 (context) succeeded: {speaker}")
            return speaker
        
        # Strategy 3: Content-type classification
        speaker = self._content_type_classification(text, text_metadata)
        if speaker:
            self.logger.debug(f"Strategy 3 (content-type) succeeded: {speaker}")
            return speaker
        
        # Strategy 4: Proximity-based assignment
        speaker = self._proximity_based_assignment(segments, target_idx, text_metadata)
        if speaker:
            self.logger.debug(f"Strategy 4 (proximity) succeeded: {speaker}")
            return speaker
        
        # Strategy 5: Statistical pattern matching
        speaker = self._statistical_pattern_matching(text, attribution_patterns)
        if speaker:
            self.logger.debug(f"Strategy 5 (statistical) succeeded: {speaker}")
            return speaker
        
        # Strategy 6: Conservative fallback
        speaker = self._conservative_fallback(text)
        if speaker:
            self.logger.debug(f"Strategy 6 (conservative) succeeded: {speaker}")
            return speaker
        
        return None
    
    def _heuristic_speaker_detection(self, text: str, text_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Heuristic speaker detection using strong patterns and indicators.
        """
        character_names = text_metadata.get('potential_character_names', set())
        
        # Look for strong dialogue attribution patterns
        attribution_patterns = [
            r'"([^"]*)",?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|asked|replied|whispered|shouted)',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|asked|replied|whispered|shouted)[^.]*\.\s*"([^"]*)"',
            r'"([^"]*)",?\s*[-–—]\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
        ]
        
        for pattern in attribution_patterns:
            match = re.search(pattern, text)
            if match:
                # Extract potential speaker (usually in group 2, sometimes group 1)
                groups = match.groups()
                for group in groups:
                    if group and len(group) > 1 and group[0].isupper():
                        # Check if it matches a known character
                        if group in character_names:
                            return group
                        elif character_names:
                            best_match = process.extractOne(group, list(character_names))
                            if best_match and best_match[1] > 85:
                                return best_match[0]
        
        # Look for possessive patterns (Character's ...)
        possessive_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\'s\s+(?:voice|words|response|question)', text)
        if possessive_match:
            potential_speaker = possessive_match.group(1)
            if potential_speaker in character_names:
                return potential_speaker
        
        return None
    
    def _context_based_recovery(self, segments: List[Dict[str, Any]], target_idx: int,
                               text_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Recover speaker using context from surrounding segments.
        """
        # Get context segments
        start_idx = max(0, target_idx - self.context_window)
        end_idx = min(len(segments), target_idx + self.context_window + 1)
        
        context_speakers = []
        for i in range(start_idx, end_idx):
            if i != target_idx:
                speaker = segments[i].get('speaker', '')
                if speaker and speaker not in ['AMBIGUOUS', 'UNFIXABLE', 'narrator']:
                    context_speakers.append(speaker)
        
        if not context_speakers:
            return None
        
        target_text = segments[target_idx].get('text', '')
        
        # If this looks like dialogue and we have recent speakers, try conversation flow
        if self._is_likely_dialogue(target_text):
            # Look for conversation patterns
            recent_speakers = []
            for i in range(max(0, target_idx - 2), target_idx):
                speaker = segments[i].get('speaker', '')
                if speaker and speaker not in ['AMBIGUOUS', 'UNFIXABLE', 'narrator']:
                    recent_speakers.append(speaker)
            
            # Simple alternation pattern
            if len(recent_speakers) >= 1:
                last_speaker = recent_speakers[-1]
                other_speakers = [s for s in context_speakers if s != last_speaker]
                if other_speakers:
                    # Prefer the most recent different speaker
                    return other_speakers[-1]
        
        # Fallback to most common speaker in context
        if context_speakers:
            speaker_counts = Counter(context_speakers)
            return speaker_counts.most_common(1)[0][0]
        
        return None
    
    def _content_type_classification(self, text: str, text_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Classify content type and assign appropriate speaker.
        """
        # Enhanced dialogue detection
        dialogue_score = 0
        narrative_score = 0
        
        # Count dialogue indicators
        for indicator in self.dialogue_indicators:
            dialogue_score += text.count(indicator)
        
        # Count narrative indicators
        for indicator in self.narrative_indicators:
            narrative_score += text.lower().count(indicator)
        
        # Check narrative patterns
        for pattern in self.narrative_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            narrative_score += matches * 2  # Weight pattern matches more heavily
        
        # Decision logic
        total_score = dialogue_score + narrative_score
        if total_score == 0:
            return None
        
        narrative_ratio = narrative_score / total_score
        
        # Strong indicators for narrator
        if (narrative_ratio > 0.7 or 
            (narrative_ratio > 0.5 and not any(marker in text for marker in ['"', '"', '"']))):
            return 'narrator'
        
        # If it's clearly dialogue but we can't identify speaker, keep as AMBIGUOUS
        if dialogue_score > narrative_score and dialogue_score > 0:
            return 'AMBIGUOUS'
        
        return None
    
    def _proximity_based_assignment(self, segments: List[Dict[str, Any]], target_idx: int,
                                   text_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Assign speaker based on proximity to successfully attributed segments.
        """
        target_text = segments[target_idx].get('text', '')
        
        # Look for the nearest successfully attributed segment with similar content
        max_distance = 5  # Maximum distance to consider
        best_match = None
        best_similarity = 0
        
        for distance in range(1, max_distance + 1):
            # Check both directions
            for direction in [-1, 1]:
                check_idx = target_idx + (direction * distance)
                if 0 <= check_idx < len(segments):
                    segment = segments[check_idx]
                    speaker = segment.get('speaker', '')
                    
                    if speaker and speaker not in ['AMBIGUOUS', 'UNFIXABLE']:
                        # Calculate text similarity
                        segment_text = segment.get('text', '')
                        similarity = fuzz.partial_ratio(target_text.lower(), segment_text.lower())
                        
                        # Boost similarity for same content type (dialogue vs narrative)
                        target_is_dialogue = self._is_likely_dialogue(target_text)
                        segment_is_dialogue = self._is_likely_dialogue(segment_text)
                        if target_is_dialogue == segment_is_dialogue:
                            similarity += 20
                        
                        # Adjust for distance (closer is better)
                        adjusted_similarity = similarity - (distance * 5)
                        
                        if adjusted_similarity > best_similarity and adjusted_similarity > 60:
                            best_similarity = adjusted_similarity
                            best_match = speaker
        
        return best_match
    
    def _statistical_pattern_matching(self, text: str, attribution_patterns: Dict[str, Any]) -> Optional[str]:
        """
        Use statistical patterns from successful attributions to predict speaker.
        """
        if not attribution_patterns or not attribution_patterns.get('character_patterns'):
            return None
        
        character_patterns = attribution_patterns['character_patterns']
        best_match = None
        best_score = 0
        
        # Analyze text patterns against known character patterns
        for character, patterns in character_patterns.items():
            score = 0
            
            # Word pattern matching
            if 'common_words' in patterns:
                text_words = set(re.findall(r'\b\w+\b', text.lower()))
                common_words = set(patterns['common_words'])
                overlap = len(text_words & common_words)
                score += overlap * 2
            
            # Length pattern matching
            if 'avg_length' in patterns:
                length_diff = abs(len(text) - patterns['avg_length'])
                if length_diff < 50:  # Similar length
                    score += 5
            
            # Dialogue vs narrative preference
            if 'dialogue_ratio' in patterns:
                text_is_dialogue = self._is_likely_dialogue(text)
                pattern_prefers_dialogue = patterns['dialogue_ratio'] > 0.5
                if text_is_dialogue == pattern_prefers_dialogue:
                    score += 10
            
            if score > best_score and score > 15:  # Minimum threshold
                best_score = score
                best_match = character
        
        return best_match
    
    def _conservative_fallback(self, text: str) -> Optional[str]:
        """
        Conservative fallback that provides the safest possible classification.
        """
        # Very conservative dialogue detection
        strong_dialogue_indicators = ['"', '"', '"']
        if any(indicator in text for indicator in strong_dialogue_indicators):
            # Has dialogue markers but we can't identify speaker
            return 'AMBIGUOUS'
        
        # Check for strong narrative indicators
        strong_narrative_patterns = [
            r'\b(?:he|she|they)\s+(?:was|were|had|would|could)\b',
            r'\bthe\s+\w+\s+(?:walked|looked|turned|stepped|moved)\b',
            r'\b(?:was|were)\s+\w+ing\b'
        ]
        
        for pattern in strong_narrative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 'narrator'
        
        # If all else fails and it's not clearly dialogue, default to narrator
        if not any(marker in text for marker in ['"', '"', '"', "'", '—']):
            return 'narrator'
        
        return None
    
    def _build_attribution_patterns(self, segments: List[Dict[str, Any]], 
                                   text_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build statistical patterns from successfully attributed segments.
        """
        character_patterns = defaultdict(lambda: {
            'texts': [],
            'lengths': [],
            'word_counts': defaultdict(int),
            'dialogue_count': 0,
            'narrative_count': 0
        })
        
        for segment in segments:
            speaker = segment.get('speaker', '')
            text = segment.get('text', '')
            
            if speaker and speaker not in ['AMBIGUOUS', 'UNFIXABLE', 'narrator'] and text:
                pattern = character_patterns[speaker]
                pattern['texts'].append(text)
                pattern['lengths'].append(len(text))
                
                # Word frequency analysis
                words = re.findall(r'\b\w+\b', text.lower())
                for word in words:
                    pattern['word_counts'][word] += 1
                
                # Content type analysis
                if self._is_likely_dialogue(text):
                    pattern['dialogue_count'] += 1
                else:
                    pattern['narrative_count'] += 1
        
        # Convert to summary statistics
        summary_patterns = {}
        for character, pattern in character_patterns.items():
            if len(pattern['texts']) >= 2:  # Need minimum samples
                total_segments = len(pattern['texts'])
                summary_patterns[character] = {
                    'avg_length': sum(pattern['lengths']) / len(pattern['lengths']),
                    'common_words': [word for word, count in pattern['word_counts'].most_common(10)],
                    'dialogue_ratio': pattern['dialogue_count'] / total_segments,
                    'sample_count': total_segments
                }
        
        return {'character_patterns': summary_patterns}
    
    def _is_likely_dialogue(self, text: str) -> bool:
        """Check if text is likely dialogue based on markers."""
        dialogue_markers = ['"', '"', '"', "'", '—', '–']
        return any(marker in text for marker in dialogue_markers)