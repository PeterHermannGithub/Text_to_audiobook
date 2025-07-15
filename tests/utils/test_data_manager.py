"""
Test data management system for the text-to-audiobook testing framework.

Provides comprehensive test data generation, validation, and management
for unit, integration, and performance testing scenarios.
"""

import json
import random
import string
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from faker import Faker
import uuid

# Initialize Faker for realistic test data generation
fake = Faker()


@dataclass
class TestBookMetadata:
    """Metadata for generated test books."""
    title: str
    author: str
    genre: str
    chapter_count: int
    word_count: int
    character_count: int
    dialogue_percentage: float
    narrative_percentage: float
    speakers: List[str]
    complexity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'author': self.author,
            'genre': self.genre,
            'chapter_count': self.chapter_count,
            'word_count': self.word_count,
            'character_count': self.character_count,
            'dialogue_percentage': self.dialogue_percentage,
            'narrative_percentage': self.narrative_percentage,
            'speakers': self.speakers,
            'complexity_score': self.complexity_score
        }


@dataclass
class TestSegment:
    """Test segment with expected validation results."""
    segment_id: str
    text_content: str
    speaker: str
    segment_type: str
    expected_quality_score: float
    expected_confidence: float
    validation_issues: List[str] = field(default_factory=list)
    refinement_needed: bool = False
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSpeaker:
    """Test speaker with characteristics."""
    speaker_id: str
    name: str
    voice_type: str
    dialogue_count: int
    consistency_score: float
    confidence_scores: List[float]
    characteristics: Dict[str, str] = field(default_factory=dict)


class TestDataGenerator:
    """
    Comprehensive test data generator for various testing scenarios.
    
    Generates realistic book content, speakers, segments, and processing
    results for testing all components of the text-to-audiobook pipeline.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize test data generator.
        
        Args:
            seed: Random seed for reproducible test data
        """
        if seed:
            random.seed(seed)
            fake.seed_instance(seed)
        
        self.genres = [
            'fiction', 'fantasy', 'science_fiction', 'mystery', 'romance',
            'thriller', 'horror', 'adventure', 'historical', 'drama'
        ]
        
        self.voice_types = [
            'young_male', 'young_female', 'adult_male', 'adult_female',
            'elderly_male', 'elderly_female', 'child', 'narrator'
        ]
        
        self.dialogue_patterns = [
            '"{text}," {speaker} said.',
            '"{text}," {speaker} replied.',
            '"{text}," {speaker} whispered.',
            '"{text}," {speaker} shouted.',
            '"{text}!" {speaker} exclaimed.',
            '"{text}?" {speaker} asked.',
            '{speaker} said, "{text}"',
            '{speaker} replied, "{text}"'
        ]
        
        self.narrative_starters = [
            'The {character} walked through the {location}.',
            'It was a {adjective} {time_of_day} when {character} arrived.',
            'The {weather} made the {location} feel {mood}.',
            'Years had passed since {character} last visited {location}.',
            'The sound of {sound} echoed through the {location}.',
            'As {character} approached the {location}, they noticed {observation}.'
        ]
    
    def generate_book_content(self, 
                            chapter_count: int = 5,
                            target_word_count: int = 5000,
                            dialogue_ratio: float = 0.4,
                            speaker_count: int = 4) -> Tuple[str, TestBookMetadata]:
        """
        Generate realistic book content with specified characteristics.
        
        Args:
            chapter_count: Number of chapters to generate
            target_word_count: Target total word count
            dialogue_ratio: Ratio of dialogue to narrative (0.0-1.0)
            speaker_count: Number of different speakers
            
        Returns:
            Tuple of (book_content, metadata)
        """
        
        # Generate book metadata
        title = fake.catch_phrase()
        author = fake.name()
        genre = random.choice(self.genres)
        
        # Generate speakers
        speakers = self.generate_speakers(speaker_count)
        speaker_names = [s.name for s in speakers]
        
        # Calculate words per chapter
        words_per_chapter = target_word_count // chapter_count
        
        chapters = []
        total_word_count = 0
        total_dialogue_words = 0
        
        for chapter_num in range(1, chapter_count + 1):
            chapter_content, chapter_stats = self.generate_chapter_content(
                chapter_num=chapter_num,
                target_words=words_per_chapter,
                dialogue_ratio=dialogue_ratio,
                speakers=speaker_names
            )
            
            chapters.append(chapter_content)
            total_word_count += chapter_stats['word_count']
            total_dialogue_words += chapter_stats['dialogue_words']
        
        # Combine chapters
        book_content = '\n\n'.join(chapters)
        character_count = len(book_content)
        
        # Calculate final statistics
        actual_dialogue_ratio = total_dialogue_words / total_word_count if total_word_count > 0 else 0
        narrative_ratio = 1.0 - actual_dialogue_ratio
        
        # Calculate complexity score based on various factors
        complexity_score = self.calculate_complexity_score(
            book_content, speaker_count, actual_dialogue_ratio
        )
        
        metadata = TestBookMetadata(
            title=title,
            author=author,
            genre=genre,
            chapter_count=chapter_count,
            word_count=total_word_count,
            character_count=character_count,
            dialogue_percentage=actual_dialogue_ratio * 100,
            narrative_percentage=narrative_ratio * 100,
            speakers=speaker_names,
            complexity_score=complexity_score
        )
        
        return book_content, metadata
    
    def generate_speakers(self, count: int) -> List[TestSpeaker]:
        """Generate test speakers with characteristics."""
        speakers = []
        
        # Always include narrator
        narrator = TestSpeaker(
            speaker_id='narrator',
            name='Narrator',
            voice_type='narrator',
            dialogue_count=0,
            consistency_score=1.0,
            confidence_scores=[1.0],
            characteristics={
                'tone': 'neutral',
                'pace': 'moderate',
                'style': 'descriptive'
            }
        )
        speakers.append(narrator)
        
        # Generate character speakers
        for i in range(count - 1):
            speaker_id = f'character_{i + 1}'
            name = fake.first_name()
            voice_type = random.choice([vt for vt in self.voice_types if vt != 'narrator'])
            
            # Generate realistic confidence scores with some variation
            base_confidence = random.uniform(0.7, 0.95)
            confidence_scores = [
                max(0.5, min(1.0, base_confidence + random.gauss(0, 0.1)))
                for _ in range(random.randint(5, 15))
            ]
            
            consistency_score = 1.0 - (max(confidence_scores) - min(confidence_scores))
            
            speaker = TestSpeaker(
                speaker_id=speaker_id,
                name=name,
                voice_type=voice_type,
                dialogue_count=random.randint(3, 20),
                consistency_score=consistency_score,
                confidence_scores=confidence_scores,
                characteristics={
                    'age_group': voice_type.split('_')[0],
                    'gender': voice_type.split('_')[1] if '_' in voice_type else 'neutral',
                    'personality': random.choice(['friendly', 'serious', 'cheerful', 'mysterious', 'wise'])
                }
            )
            speakers.append(speaker)
        
        return speakers
    
    def generate_chapter_content(self, 
                               chapter_num: int,
                               target_words: int,
                               dialogue_ratio: float,
                               speakers: List[str]) -> Tuple[str, Dict[str, int]]:
        """Generate content for a single chapter."""
        
        chapter_title = f"Chapter {chapter_num}: {fake.catch_phrase()}"
        
        # Calculate target dialogue and narrative words
        dialogue_words_target = int(target_words * dialogue_ratio)
        narrative_words_target = target_words - dialogue_words_target
        
        content_parts = [chapter_title, ""]
        
        current_dialogue_words = 0
        current_narrative_words = 0
        
        # Generate opening narrative
        opening_narrative = self.generate_narrative_segment(50)
        content_parts.append(opening_narrative)
        current_narrative_words += len(opening_narrative.split())
        
        # Generate alternating dialogue and narrative
        while (current_dialogue_words < dialogue_words_target or 
               current_narrative_words < narrative_words_target):
            
            # Decide whether to add dialogue or narrative
            if (current_dialogue_words < dialogue_words_target and 
                (current_narrative_words >= narrative_words_target or random.random() < 0.6)):
                
                # Add dialogue exchange
                dialogue_exchange = self.generate_dialogue_exchange(speakers, 30, 80)
                content_parts.append(dialogue_exchange)
                current_dialogue_words += len(dialogue_exchange.split())
                
            else:
                # Add narrative segment
                narrative_segment = self.generate_narrative_segment(40, 100)
                content_parts.append(narrative_segment)
                current_narrative_words += len(narrative_segment.split())
        
        chapter_content = '\n\n'.join(content_parts)
        
        stats = {
            'word_count': len(chapter_content.split()),
            'dialogue_words': current_dialogue_words,
            'narrative_words': current_narrative_words
        }
        
        return chapter_content, stats
    
    def generate_dialogue_exchange(self, speakers: List[str], min_words: int = 20, max_words: int = 60) -> str:
        """Generate a dialogue exchange between speakers."""
        
        # Select 2-3 speakers for the exchange
        exchange_speakers = random.sample([s for s in speakers if s != 'Narrator'], 
                                        min(len(speakers) - 1, random.randint(2, 3)))
        
        dialogue_lines = []
        target_words = random.randint(min_words, max_words)
        current_words = 0
        
        while current_words < target_words:
            speaker = random.choice(exchange_speakers)
            
            # Generate dialogue text
            dialogue_text = fake.sentence(nb_words=random.randint(3, 12))
            
            # Choose dialogue pattern
            pattern = random.choice(self.dialogue_patterns)
            dialogue_line = pattern.format(text=dialogue_text, speaker=speaker)
            
            dialogue_lines.append(dialogue_line)
            current_words += len(dialogue_line.split())
        
        return ' '.join(dialogue_lines)
    
    def generate_narrative_segment(self, min_words: int = 30, max_words: int = 80) -> str:
        """Generate a narrative segment."""
        
        target_words = random.randint(min_words, max_words)
        
        # Start with a narrative starter
        starter_template = random.choice(self.narrative_starters)
        
        # Fill in template variables
        narrative_text = starter_template.format(
            character=fake.first_name(),
            location=fake.word(),
            adjective=fake.word(),
            time_of_day=random.choice(['morning', 'afternoon', 'evening', 'night']),
            weather=random.choice(['rain', 'sunshine', 'wind', 'snow']),
            mood=fake.word(),
            sound=fake.word(),
            observation=fake.sentence()
        )
        
        # Add additional sentences to reach target word count
        current_words = len(narrative_text.split())
        
        while current_words < target_words:
            additional_sentence = fake.sentence(nb_words=random.randint(5, 15))
            narrative_text += f" {additional_sentence}"
            current_words += len(additional_sentence.split())
        
        return narrative_text
    
    def calculate_complexity_score(self, content: str, speaker_count: int, dialogue_ratio: float) -> float:
        """Calculate complexity score for content."""
        
        base_score = 5.0
        
        # Factor in speaker count
        if speaker_count <= 2:
            speaker_factor = 1.0
        elif speaker_count <= 4:
            speaker_factor = 1.5
        else:
            speaker_factor = 2.0
        
        # Factor in dialogue ratio
        if dialogue_ratio <= 0.2:
            dialogue_factor = 1.0
        elif dialogue_ratio <= 0.5:
            dialogue_factor = 1.3
        else:
            dialogue_factor = 1.6
        
        # Factor in content length
        word_count = len(content.split())
        if word_count <= 1000:
            length_factor = 1.0
        elif word_count <= 5000:
            length_factor = 1.2
        else:
            length_factor = 1.4
        
        # Calculate final score
        complexity_score = base_score * speaker_factor * dialogue_factor * length_factor
        
        # Clamp to 1-10 range
        return max(1.0, min(10.0, complexity_score))
    
    def generate_test_segments(self, content: str, expected_speakers: List[str]) -> List[TestSegment]:
        """Generate test segments with expected validation results."""
        
        segments = []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            # Skip chapter headers
            if paragraph.startswith('Chapter'):
                continue
            
            # Determine segment type and speaker
            if '"' in paragraph and any(speaker in paragraph for speaker in expected_speakers):
                segment_type = 'dialogue'
                # Try to identify speaker from text
                speaker = 'AMBIGUOUS'
                for spkr in expected_speakers:
                    if spkr in paragraph and spkr != 'Narrator':
                        speaker = spkr
                        break
                base_quality = random.uniform(0.8, 0.95)
                base_confidence = random.uniform(0.85, 0.98)
            else:
                segment_type = 'narrative'
                speaker = 'Narrator'
                base_quality = random.uniform(0.75, 0.90)
                base_confidence = random.uniform(0.90, 0.99)
            
            # Add some variation and potential issues
            quality_score = base_quality
            confidence = base_confidence
            issues = []
            refinement_needed = False
            
            # Add realistic quality variations
            if len(paragraph) < 20:
                issues.append('segment_too_short')
                quality_score *= 0.8
                refinement_needed = True
            
            if len(paragraph) > 500:
                issues.append('segment_too_long')
                quality_score *= 0.9
                refinement_needed = True
            
            if segment_type == 'dialogue' and not any(char in paragraph for char in ['"', "'", '«', '»']):
                issues.append('missing_dialogue_markers')
                quality_score *= 0.7
                refinement_needed = True
            
            if random.random() < 0.1:  # 10% chance of ambiguous speaker
                speaker = 'AMBIGUOUS'
                confidence *= 0.6
                issues.append('ambiguous_speaker')
            
            segment = TestSegment(
                segment_id=f'test_seg_{i:04d}',
                text_content=paragraph,
                speaker=speaker,
                segment_type=segment_type,
                expected_quality_score=quality_score,
                expected_confidence=confidence,
                validation_issues=issues,
                refinement_needed=refinement_needed,
                processing_metadata={
                    'word_count': len(paragraph.split()),
                    'character_count': len(paragraph),
                    'paragraph_index': i
                }
            )
            
            segments.append(segment)
        
        return segments
    
    def generate_kafka_messages(self, job_id: str, segments: List[TestSegment], chunk_size: int = 5) -> List[Dict[str, Any]]:
        """Generate Kafka messages for test segments."""
        
        messages = []
        
        # File upload message
        file_upload_msg = {
            'message_type': 'file_upload',
            'job_id': job_id,
            'file_path': f'test_data/{job_id}.txt',
            'file_size': sum(len(seg.text_content) for seg in segments),
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'total_segments': len(segments),
                'estimated_complexity': random.uniform(5.0, 8.0)
            }
        }
        messages.append(file_upload_msg)
        
        # Text extraction message
        text_extraction_msg = {
            'message_type': 'text_extraction',
            'job_id': job_id,
            'extracted_text': '\n\n'.join(seg.text_content for seg in segments),
            'extraction_metadata': {
                'format': 'txt',
                'pages_processed': random.randint(10, 50),
                'extraction_time': random.uniform(1.0, 5.0)
            },
            'timestamp': datetime.now().isoformat()
        }
        messages.append(text_extraction_msg)
        
        # Chunk processing messages
        chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
        
        for chunk_index, chunk in enumerate(chunks):
            chunk_msg = {
                'message_type': 'chunk_processing',
                'job_id': job_id,
                'chunk_id': f'{job_id}_chunk_{chunk_index:03d}',
                'chunk_index': chunk_index,
                'total_chunks': len(chunks),
                'segments': [
                    {
                        'segment_id': seg.segment_id,
                        'text_content': seg.text_content,
                        'speaker': seg.speaker,
                        'segment_type': seg.segment_type
                    }
                    for seg in chunk
                ],
                'processing_metadata': {
                    'chunk_size': len(chunk),
                    'estimated_processing_time': len(chunk) * 0.5
                },
                'timestamp': datetime.now().isoformat()
            }
            messages.append(chunk_msg)
        
        return messages
    
    def generate_performance_test_data(self, 
                                     dataset_sizes: List[int],
                                     complexity_levels: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Generate performance test datasets of various sizes and complexities."""
        
        if complexity_levels is None:
            complexity_levels = ['simple', 'moderate', 'complex']
        
        performance_datasets = {}
        
        for size in dataset_sizes:
            for complexity in complexity_levels:
                
                # Adjust parameters based on complexity
                if complexity == 'simple':
                    speaker_count = 2
                    dialogue_ratio = 0.2
                    chapter_count = max(1, size // 2000)
                elif complexity == 'moderate':
                    speaker_count = 4
                    dialogue_ratio = 0.4
                    chapter_count = max(1, size // 1500)
                else:  # complex
                    speaker_count = 6
                    dialogue_ratio = 0.6
                    chapter_count = max(1, size // 1000)
                
                dataset_name = f'{complexity}_{size}_words'
                
                # Generate content
                content, metadata = self.generate_book_content(
                    chapter_count=chapter_count,
                    target_word_count=size,
                    dialogue_ratio=dialogue_ratio,
                    speaker_count=speaker_count
                )
                
                # Generate segments
                segments = self.generate_test_segments(content, metadata.speakers)
                
                # Generate Kafka messages
                job_id = f'perf_test_{dataset_name}_{uuid.uuid4().hex[:8]}'
                messages = self.generate_kafka_messages(job_id, segments)
                
                performance_datasets[dataset_name] = {
                    'content': content,
                    'metadata': metadata.to_dict(),
                    'segments': [
                        {
                            'segment_id': seg.segment_id,
                            'text_content': seg.text_content,
                            'speaker': seg.speaker,
                            'segment_type': seg.segment_type,
                            'expected_quality_score': seg.expected_quality_score,
                            'expected_confidence': seg.expected_confidence,
                            'validation_issues': seg.validation_issues
                        }
                        for seg in segments
                    ],
                    'kafka_messages': messages,
                    'performance_expectations': {
                        'max_processing_time_seconds': size * 0.001,  # 1ms per word
                        'min_quality_score': 0.7 if complexity == 'complex' else 0.8,
                        'max_memory_usage_mb': size * 0.01,  # 10KB per word
                        'expected_segments': len(segments)
                    }
                }
        
        return performance_datasets


class TestDataManager:
    """
    Test data manager for organizing and persisting test datasets.
    
    Manages test data lifecycle including generation, storage, retrieval,
    and cleanup for various testing scenarios.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize test data manager.
        
        Args:
            base_path: Base directory for test data storage
        """
        self.base_path = base_path or Path(tempfile.gettempdir()) / 'text_to_audiobook_test_data'
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.generator = TestDataGenerator()
        
        # Create subdirectories
        self.datasets_path = self.base_path / 'datasets'
        self.fixtures_path = self.base_path / 'fixtures'
        self.performance_path = self.base_path / 'performance'
        self.cache_path = self.base_path / 'cache'
        
        for path in [self.datasets_path, self.fixtures_path, self.performance_path, self.cache_path]:
            path.mkdir(exist_ok=True)
        
        # Load or create test data inventory
        self.inventory_file = self.base_path / 'inventory.json'
        self.inventory = self.load_inventory()
    
    def load_inventory(self) -> Dict[str, Any]:
        """Load test data inventory."""
        if self.inventory_file.exists():
            with open(self.inventory_file, 'r') as f:
                return json.load(f)
        return {
            'datasets': {},
            'fixtures': {},
            'performance_data': {},
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def save_inventory(self):
        """Save test data inventory."""
        self.inventory['last_updated'] = datetime.now().isoformat()
        with open(self.inventory_file, 'w') as f:
            json.dump(self.inventory, f, indent=2)
    
    def create_test_dataset(self, 
                          name: str,
                          word_count: int = 2000,
                          complexity: str = 'moderate',
                          speakers: int = 3,
                          force_recreate: bool = False) -> Dict[str, Any]:
        """
        Create or retrieve a test dataset.
        
        Args:
            name: Dataset name
            word_count: Target word count
            complexity: Complexity level (simple/moderate/complex)
            speakers: Number of speakers
            force_recreate: Force recreation if dataset exists
            
        Returns:
            Dataset dictionary with content, metadata, and segments
        """
        
        if name in self.inventory['datasets'] and not force_recreate:
            return self.load_dataset(name)
        
        # Map complexity to parameters
        complexity_params = {
            'simple': {'dialogue_ratio': 0.2, 'chapter_count': max(1, word_count // 2000)},
            'moderate': {'dialogue_ratio': 0.4, 'chapter_count': max(1, word_count // 1500)},
            'complex': {'dialogue_ratio': 0.6, 'chapter_count': max(1, word_count // 1000)}
        }
        
        params = complexity_params.get(complexity, complexity_params['moderate'])
        
        # Generate content
        content, metadata = self.generator.generate_book_content(
            chapter_count=params['chapter_count'],
            target_word_count=word_count,
            dialogue_ratio=params['dialogue_ratio'],
            speaker_count=speakers
        )
        
        # Generate segments
        segments = self.generator.generate_test_segments(content, metadata.speakers)
        
        # Create dataset
        dataset = {
            'name': name,
            'content': content,
            'metadata': metadata.to_dict(),
            'segments': [
                {
                    'segment_id': seg.segment_id,
                    'text_content': seg.text_content,
                    'speaker': seg.speaker,
                    'segment_type': seg.segment_type,
                    'expected_quality_score': seg.expected_quality_score,
                    'expected_confidence': seg.expected_confidence,
                    'validation_issues': seg.validation_issues,
                    'refinement_needed': seg.refinement_needed,
                    'processing_metadata': seg.processing_metadata
                }
                for seg in segments
            ],
            'test_speakers': [
                {
                    'speaker_id': speaker.speaker_id,
                    'name': speaker.name,
                    'voice_type': speaker.voice_type,
                    'dialogue_count': speaker.dialogue_count,
                    'consistency_score': speaker.consistency_score,
                    'confidence_scores': speaker.confidence_scores,
                    'characteristics': speaker.characteristics
                }
                for speaker in self.generator.generate_speakers(speakers)
            ],
            'created_at': datetime.now().isoformat(),
            'parameters': {
                'word_count': word_count,
                'complexity': complexity,
                'speakers': speakers,
                'dialogue_ratio': params['dialogue_ratio']
            }
        }
        
        # Save dataset
        dataset_file = self.datasets_path / f'{name}.json'
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Update inventory
        self.inventory['datasets'][name] = {
            'file': str(dataset_file),
            'parameters': dataset['parameters'],
            'created_at': dataset['created_at'],
            'word_count': word_count,
            'segment_count': len(segments)
        }
        self.save_inventory()
        
        return dataset
    
    def load_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a test dataset by name."""
        if name not in self.inventory['datasets']:
            return None
        
        dataset_info = self.inventory['datasets'][name]
        dataset_file = Path(dataset_info['file'])
        
        if not dataset_file.exists():
            return None
        
        with open(dataset_file, 'r') as f:
            return json.load(f)
    
    def create_performance_datasets(self, 
                                  sizes: List[int] = None,
                                  complexities: List[str] = None) -> Dict[str, Any]:
        """Create performance test datasets."""
        
        if sizes is None:
            sizes = [500, 1000, 2500, 5000, 10000]
        
        if complexities is None:
            complexities = ['simple', 'moderate', 'complex']
        
        performance_data = self.generator.generate_performance_test_data(sizes, complexities)
        
        # Save performance datasets
        performance_file = self.performance_path / 'performance_datasets.json'
        with open(performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Update inventory
        self.inventory['performance_data'] = {
            'file': str(performance_file),
            'datasets': list(performance_data.keys()),
            'created_at': datetime.now().isoformat(),
            'sizes': sizes,
            'complexities': complexities
        }
        self.save_inventory()
        
        return performance_data
    
    def get_fixture_data(self, fixture_name: str) -> Optional[Dict[str, Any]]:
        """Get predefined fixture data."""
        
        fixtures = {
            'simple_dialogue': {
                'segments': [
                    TestSegment(
                        segment_id='simple_001',
                        text_content='"Hello, how are you?" Alice asked.',
                        speaker='Alice',
                        segment_type='dialogue',
                        expected_quality_score=0.92,
                        expected_confidence=0.95
                    ),
                    TestSegment(
                        segment_id='simple_002',
                        text_content='"I\'m doing well, thank you," Bob replied.',
                        speaker='Bob',
                        segment_type='dialogue',
                        expected_quality_score=0.90,
                        expected_confidence=0.93
                    )
                ]
            },
            'complex_narrative': {
                'segments': [
                    TestSegment(
                        segment_id='complex_001',
                        text_content='The ancient castle stood majestically against the stormy sky, its towers reaching toward the dark clouds that gathered ominously overhead.',
                        speaker='Narrator',
                        segment_type='narrative',
                        expected_quality_score=0.88,
                        expected_confidence=0.97
                    )
                ]
            },
            'ambiguous_speakers': {
                'segments': [
                    TestSegment(
                        segment_id='ambiguous_001',
                        text_content='"Who goes there?" called out a voice from the shadows.',
                        speaker='AMBIGUOUS',
                        segment_type='dialogue',
                        expected_quality_score=0.65,
                        expected_confidence=0.45,
                        validation_issues=['ambiguous_speaker']
                    )
                ]
            }
        }
        
        return fixtures.get(fixture_name)
    
    def cleanup_old_data(self, max_age_days: int = 7):
        """Clean up old test data files."""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        cleaned_files = []
        
        for dataset_name, dataset_info in list(self.inventory['datasets'].items()):
            created_at = datetime.fromisoformat(dataset_info['created_at'])
            
            if created_at < cutoff_date:
                dataset_file = Path(dataset_info['file'])
                if dataset_file.exists():
                    dataset_file.unlink()
                    cleaned_files.append(str(dataset_file))
                
                del self.inventory['datasets'][dataset_name]
        
        # Clean up performance data
        if 'performance_data' in self.inventory:
            perf_info = self.inventory['performance_data']
            if 'created_at' in perf_info:
                created_at = datetime.fromisoformat(perf_info['created_at'])
                if created_at < cutoff_date:
                    perf_file = Path(perf_info['file'])
                    if perf_file.exists():
                        perf_file.unlink()
                        cleaned_files.append(str(perf_file))
                    
                    del self.inventory['performance_data']
        
        if cleaned_files:
            self.save_inventory()
        
        return cleaned_files
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of available test data."""
        
        return {
            'datasets_count': len(self.inventory['datasets']),
            'performance_datasets': 'performance_data' in self.inventory,
            'base_path': str(self.base_path),
            'total_size_mb': sum(
                Path(info['file']).stat().st_size
                for info in self.inventory['datasets'].values()
                if Path(info['file']).exists()
            ) / (1024 * 1024),
            'last_updated': self.inventory['last_updated'],
            'available_datasets': list(self.inventory['datasets'].keys())
        }


# Global test data manager instance
_global_test_data_manager = None


def get_test_data_manager() -> TestDataManager:
    """Get the global test data manager instance."""
    global _global_test_data_manager
    
    if _global_test_data_manager is None:
        _global_test_data_manager = TestDataManager()
    
    return _global_test_data_manager


def cleanup_test_data():
    """Clean up test data at module level."""
    manager = get_test_data_manager()
    manager.cleanup_old_data()


# Convenience functions
def generate_test_book(name: str, word_count: int = 2000, complexity: str = 'moderate') -> Dict[str, Any]:
    """Generate a test book with specified parameters."""
    manager = get_test_data_manager()
    return manager.create_test_dataset(name, word_count, complexity)


def get_performance_data() -> Dict[str, Any]:
    """Get or create performance test datasets."""
    manager = get_test_data_manager()
    return manager.create_performance_datasets()


def get_fixture(fixture_name: str) -> Optional[Dict[str, Any]]:
    """Get predefined fixture data."""
    manager = get_test_data_manager()
    return manager.get_fixture_data(fixture_name)