import boto3
import os
import atexit
import signal
import json
from datetime import datetime

# Global variable to track if training completed successfully
training_completed = False

def upload_training_status(status, salida_prefix, additional_info=None):
    """
    Upload a status file to S3 to indicate training status
    
    Args:
        status: Status string ('completed', 'interrupted', etc.)
        salida_prefix: S3 prefix path
        additional_info: Any additional information to include
    """
    s3_client = boto3.client('s3')
    bucket_name = 'avedian-ml'
    
    status_info = {
        'status': status,
        'timestamp': datetime.now().isoformat(),
    }
    
    if additional_info:
        status_info.update(additional_info)
        
    # Create a temporary status file
    status_file = 'training_status.json'
    with open(status_file, 'w') as f:
        json.dump(status_info, f, indent=2)
    
    # Upload the status file
    s3_path = f'stylegan/{salida_prefix}/training_status.json'
    s3_client.upload_file(status_file, bucket_name, s3_path)
    
    # Clean up
    os.remove(status_file)
    print(f"Training status '{status}' uploaded to S3")

def subir_s3(version, salida_prefix='prueba1'):
    """
    Upload files to S3 bucket
    
    Args:
        version: Version identifier for the files
        salida_prefix: S3 prefix path (default: 'prueba1')
    """
    s3_client = boto3.client('s3')
    folder_path = 'salida'
    bucket_name = 'avedian-ml'
    s3_prefix = f'stylegan/{salida_prefix}/{version}'
    
    # Count total files for logging progress
    total_files = sum([len(files) for _, _, files in os.walk(folder_path)])
    uploaded_files = 0
    
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            local_file = os.path.join(subdir, file)
            # Create the full S3 path
            relative_path = os.path.relpath(local_file, folder_path)
            s3_path = os.path.join(s3_prefix, relative_path)
            
            # Upload the file
            try:
                s3_client.upload_file(local_file, bucket_name, s3_path)
                uploaded_files += 1
                if uploaded_files % 10 == 0 or uploaded_files == total_files:
                    print(f"Uploaded {uploaded_files}/{total_files} files to {s3_prefix}")
            except Exception as e:
                print(f"Error uploading {local_file} to {s3_path}: {e}")

def handle_exit(salida_prefix):
    """Function to run on exit/termination to upload final status"""
    global training_completed
    
    if not training_completed:
        print("Training was interrupted. Uploading latest results and status...")
        
        # Try to find the latest version by checking the salida directory
        folder_path = 'salida'
        if os.path.exists(folder_path):
            # Try to determine the latest version from file timestamps
            latest_time = 0
            latest_file = None
            
            for subdir, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(subdir, file)
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
            
            if latest_file:
                # Use timestamp as version for interrupted training
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                version = f"interrupted_{timestamp}"
                
                # Upload the latest files
                subir_s3(version, salida_prefix)
                upload_training_status('interrupted', salida_prefix, 
                                       {'latest_file': latest_file, 'timestamp': timestamp})

def register_exit_handlers(salida_prefix):
    """Register exit handlers to upload status on interruption"""
    def exit_handler():
        handle_exit(salida_prefix)
    
    def signal_handler(sig, frame):
        print(f"Received signal {sig}")
        handle_exit(salida_prefix)
        os._exit(1)
    
    # Register the exit handler
    atexit.register(exit_handler)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def mark_training_completed():
    """Mark training as successfully completed"""
    global training_completed
    training_completed = True

def upload_final_results(salida_prefix):
    """Upload final results after successful training completion"""
    print("Training completed successfully. Uploading final results...")
    
    # Mark training as completed
    mark_training_completed()
    
    # Use timestamp as version for the final state
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version = f"completed_{timestamp}"
    
    # Upload final results
    subir_s3(version, salida_prefix)
    
    # Upload completion status
    upload_training_status('completed', salida_prefix, {'final_version': version})
    
    print("Final results uploaded successfully!")