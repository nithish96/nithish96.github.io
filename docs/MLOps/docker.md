### Docker

  
Docker is a containerization platform that enables the development, deployment, and scaling of applications in isolated, lightweight containers. The key components of Docker include:

###  **Docker Engine**

The core of Docker that runs and manages containers. This Consists of a daemon process (dockerd) and a command-line interface (CLI) tool (docker).

- `docker --version`: Display Docker version.
- `docker info`: Display system-wide information.
- `docker help`: Display help information.

###  **Images**

A lightweight, standalone, and executable package that includes everything needed to run a piece of software, such as the code, runtime, libraries, and system tools. Images are built from a Dockerfile and can be shared and versioned through Docker registries.

- `docker images`: List all images.
- `docker pull <image>`: Download an image from a repository.
- `docker build -t <tag> .`: Build an image from the current directory's Dockerfile.

###  **Containers**

An instance of a Docker image that runs as a process on the host machine. Containers are isolated from each other and the host system, providing consistency across different environments.

- `docker ps`: List running containers.
- `docker ps -a`: List all containers (running and stopped).
- `docker run <image>`: Create and start a container from an image.
- `docker exec -it <container> <command>`: Execute a command in a running container.

###  **Networking**

Enables communication between Docker containers and between containers and the host system.  Docker provides various network drivers, and users can create custom networks to isolate and control traffic.

- `docker network ls`: List all networks.
- `docker network create <name>`: Create a network.
- `docker run --network=<network> <image>`: Run a container in a specific network.

###  **Volumes**

A persistent data storage mechanism in Docker. Volumes can be used to share data between containers, persist data beyond the lifecycle of a container, and facilitate easy data backup and restore. 

- `docker volume ls`: List all volumes.
- `docker volume create <name>`: Create a volume.
- `docker run -v <volume>:<container_path> <image>`: Mount a volume to a container.

###  **Compose**

A tool for defining and managing multi-container Docker applications using a YAML file (`docker-compose.yml`). Allows you to define the services, networks, and volumes required for an application, making it easier to manage complex deployments.

- `docker-compose up`: Start services defined in `docker-compose.yml`.
- `docker-compose down`: Stop and remove containers defined in `docker-compose.yml`.

###  **Dockerfile Directives**

A text file containing instructions for building a Docker image.  Defines the base image, adds application code, specifies dependencies, and configures runtime settings.
- `FROM`: Base image for building.
- `COPY`: Copy files or directories into the image.
- `RUN`: Execute commands in the image.
- `EXPOSE`: Specify ports to expose.
- `CMD`: Default command to run when the container starts.

###  **Docker Registry**

A centralized repository for storing and sharing Docker images. Examples include Docker Hub, a public registry, and private registries that organizations set up for their specific needs.

- `docker login`: Log in to a Docker registry.
- `docker push <image>`: Push an image to a registry.
- `docker pull <registry>/<image>`: Pull an image from a registry.

###  **Cleanup**

- `docker system prune`: Remove all stopped containers, unused networks, and dangling images.
- `docker volume prune`: Remove all unused volumes.

###  **Docker Hub**

A cloud-based registry service provided by Docker for sharing and managing Docker images. Developers can push and pull images to and from Docker Hub, making it a central hub for the Docker community.


### References 

1. [Docker cheatsheet](https://docs.docker.com/get-started/docker_cheatsheet.pdf)
2. [The Ultimate Docker Cheat Sheet](https://dockerlabs.collabnix.com/docker/cheatsheet/)