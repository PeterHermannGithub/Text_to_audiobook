import json
from tqdm import tqdm
from .llm_orchestrator import LLMOrchestrator

class EmotionAnnotator:
    """Annotates structured text with emotional information using an LLM."""

    def __init__(self, llm_orchestrator):
        self.llm_orchestrator = llm_orchestrator

    def annotate_emotions(self, structured_text):
        """
        Adds emotional annotations to each segment of the structured text.
        """
        print("\nAnnotating text with emotions...")
        annotated_text = []
        for i, segment in enumerate(tqdm(structured_text, desc="Annotating Emotions")):
            prompt = self.llm_orchestrator.prompt_factory.create_emotion_annotation_prompt(segment['text'])
            response_text = self.llm_orchestrator._get_llm_response(prompt)
            
            # Debugging: Print raw LLM response
            print(f"\nRaw LLM response for segment {i}: {response_text}")

            emotion_data = self._parse_emotion_response(response_text, segment['text'], prompt)
            
            # Add the emotion data to the segment
            segment.update(emotion_data)
            annotated_text.append(segment)
            
        return annotated_text

    def _parse_emotion_response(self, response_text, original_text, prompt):
        """
        Parses the LLM's response to extract emotional parameters.
        """
        try:
            # The prompt asks for a JSON object, so we parse it directly
            emotion_data = json.loads(response_text)
            return {
                "emotion_label": emotion_data.get("emotion", "neutral"),
                "pitch": emotion_data.get("pitch", 0.0),
                "speaking_rate": emotion_data.get("speaking_rate", 1.0)
            }
        except (json.JSONDecodeError, TypeError) as e:
            error_message = f"Failed to parse emotion LLM response as JSON: {e}"
            self.llm_orchestrator._log_error(error_message, prompt, response_text)
            return {
                "emotion_label": "neutral",
                "pitch": 0.0,
                "speaking_rate": 1.0
            }

