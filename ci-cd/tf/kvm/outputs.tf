output "app_floating_ip" {
  description = "Public IP for the summarizer VM"
  value       = openstack_compute_instance_v2.app.floating_ip
}