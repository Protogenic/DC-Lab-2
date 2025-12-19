#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <omp.h>

const double G = 6.67430e-11;

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

struct Acceleration {
    double ax, ay, az;
};

std::vector<Particle> readParticles(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    int n;
    file >> n;
    std::vector<Particle> particles(n);
    
    for (int i = 0; i < n; i++) {
        file >> particles[i].x >> particles[i].y >> particles[i].z
             >> particles[i].vx >> particles[i].vy >> particles[i].vz;
        particles[i].mass = 1.0e24;
    }
    
    file.close();
    return particles;
}

void writeState(std::ofstream& out, double t, const std::vector<Particle>& particles) {
    out << t;
    for (const auto& p : particles) {
        out << " " << p.x << " " << p.y;
    }
    out << "\n";
}

void computeAccelerations(const std::vector<Particle>& particles, 
                          std::vector<Acceleration>& accelerations) {
    int n = particles.size();
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        accelerations[i].ax = 0.0;
        accelerations[i].ay = 0.0;
        accelerations[i].az = 0.0;
    }
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            
            double dist2 = dx*dx + dy*dy + dz*dz;
            double dist = sqrt(dist2);
            
            if (dist < 1e-10) continue;
            
            double force = G * particles[j].mass / (dist2 * dist);
            
            ax += force * dx;
            ay += force * dy;
            az += force * dz;
        }
        
        accelerations[i].ax = ax;
        accelerations[i].ay = ay;
        accelerations[i].az = az;
    }
}

void verletStep(std::vector<Particle>& particles, double dt) {
    int n = particles.size();
    std::vector<Acceleration> accelerations(n);
    
    computeAccelerations(particles, accelerations);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        particles[i].x += particles[i].vx * dt + 0.5 * accelerations[i].ax * dt * dt;
        particles[i].y += particles[i].vy * dt + 0.5 * accelerations[i].ay * dt * dt;
        particles[i].z += particles[i].vz * dt + 0.5 * accelerations[i].az * dt * dt;
        
        particles[i].vx += 0.5 * accelerations[i].ax * dt;
        particles[i].vy += 0.5 * accelerations[i].ay * dt;
        particles[i].vz += 0.5 * accelerations[i].az * dt;
    }
    
    computeAccelerations(particles, accelerations);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        particles[i].vx += 0.5 * accelerations[i].ax * dt;
        particles[i].vy += 0.5 * accelerations[i].ay * dt;
        particles[i].vz += 0.5 * accelerations[i].az * dt;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <tend> <input_file>" << std::endl;
        return 1;
    }
    
    double tend = std::atof(argv[1]);
    std::string filename = argv[2];
    
    std::vector<Particle> particles = readParticles(filename);
    int n = particles.size();
    
    std::cout << "N-Body Simulation (OpenMP)" << std::endl;
    std::cout << "Particles: " << n << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    
    double dt = 0.01;
    int num_steps = static_cast<int>(tend / dt);
    int output_every = std::max(1, num_steps / 1000);
    
    std::ofstream outfile("output.csv");
    outfile.precision(10);
    writeState(outfile, 0.0, particles);
    
    double start_time = omp_get_wtime();
    
    for (int step = 1; step <= num_steps; step++) {
        verletStep(particles, dt);
        
        if (step % output_every == 0) {
            writeState(outfile, step * dt, particles);
        }
        
        if (step % (num_steps / 10) == 0) {
            std::cout << "Progress: " << (100.0 * step / num_steps) << "%" << std::endl;
        }
    }
    
    double elapsed = omp_get_wtime() - start_time;
    outfile.close();
    
    std::cout << "Time: " << elapsed << " sec" << std::endl;
    std::cout << "Output: output.csv" << std::endl;
    
    return 0;
}