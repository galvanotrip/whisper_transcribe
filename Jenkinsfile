pipeline {
    agent any

    environment {
        // persistent model cache: survives container re-creation
        HF_HOME = '/var/jenkins_home/hf-cache'
    }

    // parameters {
    //     string(name: 'URL', defaultValue: '', description: 'Enter the video URL to process')
    // }

    stages {
        stage('🧪 Setup Python environment') {
            steps {
                sh '''
                    echo "🔧 Creating virtual environment..."
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('🚀 Run script') {
            steps {
                sh '''
                    echo "▶️ Running script with URL: $URL"
                    . venv/bin/activate
                    python -u main.py /var/smb/
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Script executed successfully.'
        }
        failure {
            echo '❌ Script execution failed. Check logs.'
        }
        always {
            echo 'ℹ️ Pipeline finished.'
        }
    }
}
