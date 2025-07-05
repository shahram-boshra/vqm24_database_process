# Use a base image with Miniconda pre-installed
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml file into the container
COPY environment.yml .

# IMPORTANT: Configure Conda for improved download robustness *before* any major Conda operations.
# This ensures these timeouts are in effect for the initial repodata.json fetches.
RUN conda config --set remote_connect_timeout_secs 60.0 && \
    conda config --set remote_read_timeout_secs 300.0 && \
    conda clean --all -y

# Update conda to the latest version. It should benefit from the above timeout settings.
RUN conda update -n base -c defaults conda -y

# Use a retry loop for creating the environment from environment.yml.
# This is crucial for large downloads and intermittent network issues.
# It will try up to 3 times, waiting 10 seconds between attempts.
RUN for i in $(seq 1 3); do \
        conda env create -f environment.yml && break; \
        echo "Conda environment creation failed, retrying in 10 seconds... (Attempt $i/3)"; \
        sleep 10; \
    done && \
    # Clean up conda cache after environment creation to reduce image size
    conda clean --all -y

# Ensure bash is the default shell for subsequent RUN instructions if it wasn't already.
# This ensures shell scripts work as expected.
SHELL ["/bin/bash", "-c"]

# Manually add the Conda base initialization to .bashrc
# This typically looks something like:
# . /opt/conda/etc/profile.d/conda.sh
# and then initializes the base environment.
# We'll then add the specific environment activation.
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate shah_env" >> ~/.bashrc

# --- IMPORTANT CHANGE ENDS HERE ---

# Copy your application code into the container
# This should happen AFTER the environment is set up, so changes to code don't invalidate the env layer cache.
COPY . /app

# Define the default command to run when the container starts
# For an interactive shell that auto-activates Conda
CMD ["/bin/bash", "--login"]
