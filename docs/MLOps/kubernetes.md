Kubernetes, often abbreviated as K8s, is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. Originally developed by Google and later donated to the Cloud Native Computing Foundation (CNCF), Kubernetes has become a standard for deploying and managing applications in containers.

###  Key Kubernetes Concepts

- **Nodes:** The physical or virtual machines that run your containers. Each node must have a container runtime (like Docker) installed.
- **Pods:** The smallest deployable units in Kubernetes. Pods can contain one or more containers that share network and storage resources.
- **Services:** An abstraction to expose a group of pods as a network service. Services enable communication between different parts of an application.
- **ReplicaSets:** Ensure a specified number of pod replicas are running at all times. It's a way to achieve high availability and scalability.
- **Deployments:** Provide declarative updates to applications. They describe the desired state for applications and manage the deployment process to achieve that state.
- **ConfigMaps and Secrets:** Store configuration information and sensitive data separately from the application code.

### Pods

  
Pod is the smallest deployable unit and the basic building block for running applications. A Pod represents a single instance of a running process in a cluster and encapsulates one or more containers. Containers within a Pod share the same network namespace, allowing them to communicate with each other using `localhost`.

- Create a Pod

    `kubectl create pod <pod-name> --image=<container-image>`
    
- List Pods

    `kubectl get pods`
    
- Describe a Pod
    
    `kubectl describe pod <pod-name>`
    
- Delete a Pod
    
    `kubectl delete pod <pod-name>`
    
- Access Pod Terminal
    
    `kubectl exec -it <pod-name> -- /bin/bash`
    
- View Pod Logs
    
    `kubectl logs <pod-name>`
    
### Deployments

  
 Deployment is an abstraction that provides declarative updates to applications. It allows you to describe the desired state for your application, and Kubernetes will then work to ensure that the current state of the deployed application matches the specified desired state. Deployments are a higher-level abstraction compared to managing individual Pods, providing features like scaling, rolling updates, and self-healing.

- Create a Deployment
    
    `kubectl create deployment <deployment-name> --image=<container-image>`
    
- List Deployments

    `kubectl get deployments`
    
- Scale a Deployment
    
    `kubectl scale deployment <deployment-name> --replicas=<num-replicas>`
    

### Services

Service is an abstraction that defines a logical set of Pods and a policy by which to access them. Services allow applications to communicate with each other or with external clients, abstracting away the details of the underlying network. 

- Expose a Deployment as a Service
    
    `kubectl expose deployment <deployment-name> --port=<port> --target-port=<target-port> --type=NodePort`
    
- List Services
    
    `kubectl get services`
    
- Describe a Service
    
    `kubectl describe service <service-name>`
    

### ConfigMaps

ConfigMaps provide a way to decouple configuration data from the application code. They allow you to manage configuration settings for your applications separately from the application code, making it easier to update configurations without modifying the application itself. ConfigMaps are particularly useful for applications that need configuration information such as environment variables, command-line arguments, or configuration files. 

- Create a ConfigMap
    
    `kubectl create configmap <configmap-name> --from-literal=<key1>=<value1> --from-literal=<key2>=<value2>`
    
- List ConfigMaps
    
    `kubectl get configmaps`
    

### Secrets

Secrets are used to store sensitive information, such as passwords, API keys, and other confidential data. Like ConfigMaps, Secrets allow you to decouple sensitive information from your application code, enhancing security and making it easier to manage and update credentials without modifying the application itself.

- Create a Secret
    
    `kubectl create secret generic <secret-name> --from-literal=<key1>=<value1> --from-literal=<key2>=<value2>`
    
- List Secrets
    
    `kubectl get secrets`

### Namespaces

Namespace is a way to organize and segregate resources within a cluster. It provides a scope for names, avoiding naming collisions between resources in different namespaces. Namespaces can be particularly useful in large, multi-tenant clusters where multiple teams or applications share the same Kubernetes cluster.

- Create a Namespace
    
    `kubectl create namespace <namespace-name>`
    
- List Namespaces
    
    `kubectl get namespaces`
    

### Context

Context is a set of access parameters for a Kubernetes cluster. It includes information such as the cluster endpoint, the user's credentials, and the namespace. Contexts allow users to switch between different clusters and configurations easily. The `kubectl` command-line tool uses contexts to determine which cluster it should interact with.


- Switch Context
    
    `kubectl config use-context <context-name>`
    

### Upgrades and Rollbacks

Upgrades and rollbacks are crucial operations to maintain the health and reliability of your cluster. Upgrading Kubernetes involves updating the control plane components, such as the API server, controller manager, scheduler, and others. Rollbacks, on the other hand, allow you to revert to a previous version of the control plane in case of issues or unexpected behavior.

- Update container image

	 `kubectl set image deployment/<deployment_name> <container_name>=<new_image>` .

- View rollout history.

	`kubectl rollout history deployment/<name>` 

- Rollback to previous deployment.

	`kubectl rollout undo deployment/<name>` 


###  Monitoring and Debugging

Monitoring and debugging are critical aspects of managing a Kubernetes cluster. They help ensure the health, performance, and reliability of applications running within the cluster. Kubernetes provides various tools and approaches for monitoring and debugging.

- Display resource usage.
    
    `kubectl top nodes/pods` 

 - Describe a resource for troubleshooting.
 
	 `kubectl describe <resource_type> <resource_name>` 

###  Persistent Volumes (PV)

Persistent Volumes (PVs) are a way to provide durable storage for applications. PVs are part of the cluster's storage infrastructure, and they represent physical or networked storage resources in the cluster. They decouple storage provisioning from Pod lifecycle, allowing for the persistence of data even when Pods are rescheduled or recreated.

- List persistent volumes.   

	 `kubectl get pv` 

- Describe a persistent volume.
    
	 `kubectl describe pv/<pv_name>` 

###  Custom Resources

Custom Resources (CRs) and Custom Resource Definitions (CRDs) provide a way to extend the Kubernetes API and introduce custom objects with specific behaviors and semantics. Custom Resources allow users to define and manage application-specific resources beyond the built-in types like Pods, Services, and Deployments.

- List Custom Resource Definitions.

	`kubectl get crd` 
		
- List instances of a custom resource.

	`kubectl get <custom_resource>` 

- Describe a custom resource instance.
    
	 `kubectl describe <custom_resource> <name>` 


### References 
1. [ kubectl Quick Reference](https://kubernetes.io/docs/reference/kubectl/quick-reference/)
2. [kubectl cheatsheet](https://www.bluematador.com/learn/kubectl-cheatsheet)