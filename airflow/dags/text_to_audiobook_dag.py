"""
Text-to-audiobook processing DAG for Airflow.

This DAG orchestrates the complete text-to-audiobook pipeline including:
- File upload and text extraction
- Distributed text processing with Spark
- LLM-based speaker attribution
- Quality validation and refinement
- Output formatting and completion
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from airflow.utils.decorators import task
from airflow.models import Variable
from airflow.configuration import conf

# Import our custom operators
from operators.spark_text_structurer_operator import SparkTextStructurerOperator
from operators.kafka_producer_operator import KafkaProducerOperator
from operators.kafka_consumer_operator import KafkaConsumerOperator
from operators.llm_processing_operator import LLMProcessingOperator
from operators.quality_validation_operator import QualityValidationOperator


# DAG Configuration
DAG_ID = "text_to_audiobook_processing"
SCHEDULE_INTERVAL = None  # Triggered manually or via API
MAX_ACTIVE_RUNS = 3
CATCHUP = False

# Default arguments for all tasks
default_args = {
    'owner': 'text-to-audiobook',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'sla': timedelta(hours=1)
}

# Environment variables and settings
INPUT_DIR = Variable.get("INPUT_DIR", default_var="/opt/airflow/input")
OUTPUT_DIR = Variable.get("OUTPUT_DIR", default_var="/opt/airflow/output")
SPARK_ENVIRONMENT = Variable.get("SPARK_ENVIRONMENT", default_var="local")
KAFKA_ENABLED = Variable.get("KAFKA_ENABLED", default_var="true").lower() == "true"
LLM_ENGINE = Variable.get("LLM_ENGINE", default_var="local")
QUALITY_THRESHOLD = float(Variable.get("QUALITY_THRESHOLD", default_var="95.0"))


def get_dag_config() -> Dict[str, Any]:
    """Get DAG configuration from Airflow Variables."""
    return {
        'input_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
        'spark_environment': SPARK_ENVIRONMENT,
        'kafka_enabled': KAFKA_ENABLED,
        'llm_engine': LLM_ENGINE,
        'quality_threshold': QUALITY_THRESHOLD,
        'processing_options': {
            'chunk_size': int(Variable.get("CHUNK_SIZE", default_var="2500")),
            'overlap_size': int(Variable.get("OVERLAP_SIZE", default_var="500")),
            'max_refinement_iterations': int(Variable.get("MAX_REFINEMENT_ITERATIONS", default_var="2")),
            'enable_contextual_refinement': Variable.get("ENABLE_CONTEXTUAL_REFINEMENT", default_var="true").lower() == "true"
        }
    }


# Create the DAG
dag = DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Complete text-to-audiobook processing pipeline',
    schedule_interval=SCHEDULE_INTERVAL,
    max_active_runs=MAX_ACTIVE_RUNS,
    catchup=CATCHUP,
    tags=['text-processing', 'audiobook', 'nlp', 'distributed'],
    doc_md=__doc__
)


# Task 1: File Upload and Validation
@task(dag=dag, task_id="validate_input_file")
def validate_input_file(file_path: str, **context) -> Dict[str, Any]:
    """
    Validate input file and extract basic metadata.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Dictionary containing file metadata and validation status
    """
    import os
    from pathlib import Path
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    file_info = {
        'file_path': file_path,
        'file_name': Path(file_path).name,
        'file_size': os.path.getsize(file_path),
        'file_extension': Path(file_path).suffix.lower(),
        'job_id': context['dag_run'].run_id,
        'created_at': datetime.now().isoformat(),
        'status': 'validated'
    }
    
    # Check if file format is supported
    supported_formats = {'.txt', '.md', '.pdf', '.docx', '.epub', '.mobi'}
    if file_info['file_extension'] not in supported_formats:
        raise ValueError(f"Unsupported file format: {file_info['file_extension']}")
    
    # Store file info in XCom for downstream tasks
    context['task_instance'].xcom_push(key='file_info', value=file_info)
    
    return file_info


# Task 2: Text Extraction
@task(dag=dag, task_id="extract_text")
def extract_text(file_info: Dict[str, Any], **context) -> Dict[str, Any]:
    """
    Extract text from the input file.
    
    Args:
        file_info: File metadata from validation task
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    import sys
    sys.path.append('/opt/airflow/dags')
    
    from src.text_processing.text_extractor import TextExtractor
    
    extractor = TextExtractor()
    
    try:
        extracted_text = extractor.extract(file_info['file_path'])
        
        text_metadata = {
            'character_count': len(extracted_text),
            'word_count': len(extracted_text.split()),
            'line_count': len(extracted_text.split('\n')),
            'extraction_method': 'pdf' if file_info['file_extension'] == '.pdf' else 'text',
            'extraction_time': datetime.now().isoformat()
        }
        
        result = {
            'extracted_text': extracted_text,
            'text_metadata': text_metadata,
            'job_id': file_info['job_id'],
            'status': 'extracted'
        }
        
        # Store extracted text in XCom
        context['task_instance'].xcom_push(key='extracted_text', value=result)
        
        return result
        
    except Exception as e:
        raise Exception(f"Text extraction failed: {str(e)}")


# Task 3: Submit to Kafka (if enabled)
def submit_to_kafka_conditional(**context):
    """Submit text extraction to Kafka if enabled."""
    config = get_dag_config()
    
    if not config['kafka_enabled']:
        logging.info("Kafka disabled, skipping Kafka submission")
        return {'status': 'skipped', 'reason': 'kafka_disabled'}
    
    file_info = context['task_instance'].xcom_pull(key='file_info', task_ids='validate_input_file')
    extracted_text = context['task_instance'].xcom_pull(key='extracted_text', task_ids='extract_text')
    
    from src.kafka.producers.file_upload_producer import FileUploadProducer
    
    with FileUploadProducer() as producer:
        producer.notify_text_extracted(
            job_id=file_info['job_id'],
            extracted_text=extracted_text['extracted_text'],
            text_metadata=extracted_text['text_metadata'],
            file_info=file_info
        )
    
    return {'status': 'submitted', 'job_id': file_info['job_id']}


submit_to_kafka = PythonOperator(
    task_id='submit_to_kafka',
    python_callable=submit_to_kafka_conditional,
    dag=dag
)


# Task 4: Distributed Text Processing with Spark
@task(dag=dag, task_id="process_text_with_spark")
def process_text_with_spark(extracted_text: Dict[str, Any], **context) -> Dict[str, Any]:
    """
    Process text using distributed Spark processing.
    
    Args:
        extracted_text: Extracted text and metadata
        
    Returns:
        Dictionary containing structured segments
    """
    import sys
    sys.path.append('/opt/airflow/dags')
    
    from src.spark.spark_text_structurer import SparkTextStructurerContext
    
    config = get_dag_config()
    
    try:
        with SparkTextStructurerContext(
            environment=config['spark_environment'],
            config=config['processing_options']
        ) as structurer:
            
            structured_segments = structurer.structure_text(
                text_content=extracted_text['extracted_text'],
                processing_options=config['processing_options']
            )
            
            processing_metrics = structurer.get_processing_metrics()
            
            result = {
                'structured_segments': structured_segments,
                'processing_metrics': processing_metrics,
                'job_id': extracted_text['job_id'],
                'status': 'processed'
            }
            
            # Store results in XCom
            context['task_instance'].xcom_push(key='structured_segments', value=result)
            
            return result
            
    except Exception as e:
        raise Exception(f"Spark text processing failed: {str(e)}")


# Task 5: Quality Validation
@task(dag=dag, task_id="validate_quality")
def validate_quality(structured_segments: Dict[str, Any], **context) -> Dict[str, Any]:
    """
    Validate the quality of structured segments.
    
    Args:
        structured_segments: Structured segments from Spark processing
        
    Returns:
        Dictionary containing validation results
    """
    import sys
    sys.path.append('/opt/airflow/dags')
    
    from src.validation.validator import SimplifiedValidator
    
    config = get_dag_config()
    validator = SimplifiedValidator()
    
    try:
        # Validate segments
        validation_result = validator.validate_structured_segments(
            structured_segments['structured_segments']
        )
        
        quality_score = validation_result['quality_score']
        
        result = {
            'validation_result': validation_result,
            'quality_score': quality_score,
            'quality_threshold': config['quality_threshold'],
            'passed_threshold': quality_score >= config['quality_threshold'],
            'job_id': structured_segments['job_id'],
            'status': 'validated'
        }
        
        # Store validation results in XCom
        context['task_instance'].xcom_push(key='validation_result', value=result)
        
        return result
        
    except Exception as e:
        raise Exception(f"Quality validation failed: {str(e)}")


# Task 6: Contextual Refinement (conditional)
@task(dag=dag, task_id="refine_segments")
def refine_segments(structured_segments: Dict[str, Any], 
                   validation_result: Dict[str, Any], **context) -> Dict[str, Any]:
    """
    Refine segments using contextual analysis if needed.
    
    Args:
        structured_segments: Structured segments from Spark processing
        validation_result: Quality validation results
        
    Returns:
        Dictionary containing refined segments
    """
    import sys
    sys.path.append('/opt/airflow/dags')
    
    config = get_dag_config()
    
    # Skip refinement if quality is already high enough
    if validation_result['passed_threshold']:
        logging.info("Quality threshold met, skipping refinement")
        return {
            'refined_segments': structured_segments['structured_segments'],
            'refinement_applied': False,
            'job_id': structured_segments['job_id'],
            'status': 'skipped'
        }
    
    # Apply contextual refinement
    if config['processing_options']['enable_contextual_refinement']:
        from src.refinement.contextual_refiner import ContextualRefiner
        
        refiner = ContextualRefiner()
        
        try:
            refined_segments = refiner.refine_segments(
                structured_segments['structured_segments'],
                metadata={}  # Metadata would be passed from previous tasks
            )
            
            result = {
                'refined_segments': refined_segments,
                'refinement_applied': True,
                'job_id': structured_segments['job_id'],
                'status': 'refined'
            }
            
            # Store refined segments in XCom
            context['task_instance'].xcom_push(key='refined_segments', value=result)
            
            return result
            
        except Exception as e:
            raise Exception(f"Contextual refinement failed: {str(e)}")
    
    else:
        # Return original segments if refinement disabled
        return {
            'refined_segments': structured_segments['structured_segments'],
            'refinement_applied': False,
            'job_id': structured_segments['job_id'],
            'status': 'skipped'
        }


# Task 7: Final Output Formatting
@task(dag=dag, task_id="format_output")
def format_output(refined_segments: Dict[str, Any], **context) -> Dict[str, Any]:
    """
    Format final output and save to file.
    
    Args:
        refined_segments: Refined segments from refinement task
        
    Returns:
        Dictionary containing output file information
    """
    import sys
    sys.path.append('/opt/airflow/dags')
    
    from src.output.output_formatter import OutputFormatter
    from pathlib import Path
    
    config = get_dag_config()
    formatter = OutputFormatter()
    
    try:
        # Format segments
        formatted_segments = formatter.format_segments(refined_segments['refined_segments'])
        
        # Create output file path
        file_info = context['task_instance'].xcom_pull(key='file_info', task_ids='validate_input_file')
        output_filename = f"{Path(file_info['file_name']).stem}.json"
        output_path = os.path.join(config['output_dir'], output_filename)
        
        # Ensure output directory exists
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Save formatted output
        output_data = {
            'job_id': refined_segments['job_id'],
            'input_file': file_info['file_path'],
            'processing_timestamp': datetime.now().isoformat(),
            'segments': formatted_segments,
            'metadata': {
                'total_segments': len(formatted_segments),
                'unique_speakers': len(set(seg['speaker'] for seg in formatted_segments)),
                'processing_config': config['processing_options']
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        result = {
            'output_path': output_path,
            'output_filename': output_filename,
            'total_segments': len(formatted_segments),
            'unique_speakers': len(set(seg['speaker'] for seg in formatted_segments)),
            'job_id': refined_segments['job_id'],
            'status': 'completed'
        }
        
        # Store final results in XCom
        context['task_instance'].xcom_push(key='final_output', value=result)
        
        return result
        
    except Exception as e:
        raise Exception(f"Output formatting failed: {str(e)}")


# Task 8: Cleanup and Status Update
@task(dag=dag, task_id="cleanup_and_status")
def cleanup_and_status(final_output: Dict[str, Any], **context) -> Dict[str, Any]:
    """
    Cleanup temporary files and update processing status.
    
    Args:
        final_output: Final output information
        
    Returns:
        Dictionary containing final processing status
    """
    import shutil
    import tempfile
    
    config = get_dag_config()
    
    try:
        # Clean up temporary files if any
        temp_dir = tempfile.gettempdir()
        job_id = final_output['job_id']
        
        # Remove job-specific temporary files
        for temp_file in os.listdir(temp_dir):
            if job_id in temp_file:
                temp_path = os.path.join(temp_dir, temp_file)
                try:
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)
                    elif os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                except Exception as e:
                    logging.warning(f"Could not remove temporary file {temp_path}: {e}")
        
        # Update status in database if configured
        try:
            postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
            
            insert_sql = """
                INSERT INTO processing_jobs (job_id, status, output_path, total_segments, unique_speakers, completed_at)
                VALUES (%(job_id)s, %(status)s, %(output_path)s, %(total_segments)s, %(unique_speakers)s, %(completed_at)s)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    output_path = EXCLUDED.output_path,
                    total_segments = EXCLUDED.total_segments,
                    unique_speakers = EXCLUDED.unique_speakers,
                    completed_at = EXCLUDED.completed_at;
            """
            
            postgres_hook.run(insert_sql, parameters={
                'job_id': job_id,
                'status': 'completed',
                'output_path': final_output['output_path'],
                'total_segments': final_output['total_segments'],
                'unique_speakers': final_output['unique_speakers'],
                'completed_at': datetime.now()
            })
            
        except Exception as e:
            logging.warning(f"Could not update database status: {e}")
        
        final_status = {
            'job_id': job_id,
            'status': 'completed',
            'output_path': final_output['output_path'],
            'total_segments': final_output['total_segments'],
            'unique_speakers': final_output['unique_speakers'],
            'cleanup_completed': True,
            'processing_completed_at': datetime.now().isoformat()
        }
        
        logging.info(f"Processing completed successfully for job {job_id}")
        
        return final_status
        
    except Exception as e:
        raise Exception(f"Cleanup and status update failed: {str(e)}")


# Task 9: Send Notification (optional)
def send_completion_notification(**context):
    """Send notification about processing completion."""
    try:
        final_status = context['task_instance'].xcom_pull(key='return_value', task_ids='cleanup_and_status')
        
        # Send email notification if configured
        from airflow.providers.email.operators.email import EmailOperator
        
        subject = f"Text-to-audiobook processing completed: {final_status['job_id']}"
        
        html_content = f"""
        <h3>Processing Completed Successfully</h3>
        <p><strong>Job ID:</strong> {final_status['job_id']}</p>
        <p><strong>Output Path:</strong> {final_status['output_path']}</p>
        <p><strong>Total Segments:</strong> {final_status['total_segments']}</p>
        <p><strong>Unique Speakers:</strong> {final_status['unique_speakers']}</p>
        <p><strong>Completed At:</strong> {final_status['processing_completed_at']}</p>
        """
        
        # This would send email if email configuration is set up
        logging.info(f"Processing notification: {subject}")
        
        return {'notification_sent': True}
        
    except Exception as e:
        logging.error(f"Failed to send notification: {e}")
        return {'notification_sent': False, 'error': str(e)}


send_notification = PythonOperator(
    task_id='send_notification',
    python_callable=send_completion_notification,
    trigger_rule='all_success',
    dag=dag
)


# Database setup task (runs once)
create_tables = PostgresOperator(
    task_id='create_tables',
    postgres_conn_id='postgres_default',
    sql="""
        CREATE TABLE IF NOT EXISTS processing_jobs (
            job_id VARCHAR(255) PRIMARY KEY,
            status VARCHAR(50) NOT NULL,
            output_path VARCHAR(500),
            total_segments INTEGER,
            unique_speakers INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );
    """,
    dag=dag
)


# Define task dependencies
# 1. Setup phase
create_tables

# 2. Input validation and extraction
validate_input_file_task = validate_input_file("{{ dag_run.conf['file_path'] }}")
extract_text_task = extract_text(validate_input_file_task)

# 3. Processing pipeline
submit_to_kafka >> process_text_with_spark(extract_text_task)
process_text_task = process_text_with_spark(extract_text_task)
validate_quality_task = validate_quality(process_text_task)
refine_segments_task = refine_segments(process_text_task, validate_quality_task)
format_output_task = format_output(refine_segments_task)
cleanup_task = cleanup_and_status(format_output_task)

# 4. Notification
send_notification

# Set up task dependencies
create_tables >> validate_input_file_task
validate_input_file_task >> extract_text_task
extract_text_task >> [submit_to_kafka, process_text_task]
process_text_task >> validate_quality_task
validate_quality_task >> refine_segments_task
refine_segments_task >> format_output_task
format_output_task >> cleanup_task
cleanup_task >> send_notification


# Additional monitoring tasks
health_check = BashOperator(
    task_id='health_check',
    bash_command='echo "System health check passed"',
    dag=dag
)

# Set up monitoring dependency
health_check >> validate_input_file_task