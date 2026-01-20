package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"
)

const (
	projectID   = "security-route-project"
	region      = "us-central1"
	serviceName = "backend-fastapi-ia"
	imageName   = "backend-ia-fastapi"
	port        = "8000"
)

func main() {
	log.Println("🚀 Iniciando despliegue de Backend FastAPI (IA) en GCP...")

	ctx := context.Background()

	if err := deployFastAPIBackend(ctx); err != nil {
		log.Fatalf("❌ Error en el despliegue: %v", err)
	}

	log.Println("\n🎉 ¡Despliegue completado exitosamente!")
}

func deployFastAPIBackend(ctx context.Context) error {
	imageFullName := fmt.Sprintf("gcr.io/%s/%s:latest", projectID, imageName)

	log.Println("🔐 Configurando autenticación con Google Container Registry...")
	authCmd := exec.CommandContext(ctx, "gcloud", "auth", "configure-docker", "--quiet")
	authCmd.Stdout = os.Stdout
	authCmd.Stderr = os.Stderr
	if err := authCmd.Run(); err != nil {
		return fmt.Errorf("error al configurar autenticación: %w", err)
	}

	log.Println("\n🔨 Construyendo imagen Docker de FastAPI...")
	buildCmd := exec.CommandContext(ctx, "docker", "build",
		"-t", imageFullName,
		"-f", "../../Dockerfile",
		"../..",
	)
	buildCmd.Stdout = os.Stdout
	buildCmd.Stderr = os.Stderr

	if err := buildCmd.Run(); err != nil {
		return fmt.Errorf("error al construir imagen: %w", err)
	}
	log.Println("✅ Imagen construida exitosamente")

	log.Println("\n☁️  Subiendo imagen a Google Container Registry...")
	pushCmd := exec.CommandContext(ctx, "docker", "push", imageFullName)
	pushCmd.Stdout = os.Stdout
	pushCmd.Stderr = os.Stderr

	if err := pushCmd.Run(); err != nil {
		return fmt.Errorf("error al subir imagen: %w", err)
	}
	log.Println("✅ Imagen subida exitosamente")

	log.Println("\n🚢 Desplegando en Cloud Run...")
	deployCmd := exec.CommandContext(ctx, "gcloud", "run", "deploy", serviceName,
		"--image", imageFullName,
		"--platform", "managed",
		"--region", region,
		"--allow-unauthenticated",
		"--port", port,
		"--max-instances", "10",
		"--min-instances", "0",
		"--memory", "2Gi",
		"--cpu", "2",
		"--timeout", "600",
		"--project", projectID,
		"--quiet",
	)
	deployCmd.Stdout = os.Stdout
	deployCmd.Stderr = os.Stderr

	if err := deployCmd.Run(); err != nil {
		return fmt.Errorf("error al desplegar en Cloud Run: %w", err)
	}
	log.Println("✅ Servicio desplegado exitosamente")

	log.Println("\n🔗 Obteniendo URL del servicio...")
	time.Sleep(2 * time.Second)

	urlCmd := exec.CommandContext(ctx, "gcloud", "run", "services", "describe", serviceName,
		"--platform", "managed",
		"--region", region,
		"--format", "value(status.url)",
		"--project", projectID,
	)

	output, err := urlCmd.Output()
	if err != nil {
		log.Printf("⚠️  No se pudo obtener la URL automáticamente")
	} else {
		log.Printf("\n✨ URL del Backend FastAPI (IA):")
		log.Printf("   %s", string(output))
	}

	log.Println("\n📊 Información del despliegue:")
	log.Printf("   • Proyecto: %s", projectID)
	log.Printf("   • Servicio: %s", serviceName)
	log.Printf("   • Región: %s", region)
	log.Printf("   • Imagen: %s", imageFullName)
	log.Printf("   • Recursos: 2 CPU, 2GB RAM (optimizado para IA)")

	return nil
}
