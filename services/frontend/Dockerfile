FROM nginx:alpine

# Servir archivos estáticos desde la raíz del proyecto
COPY static/ /usr/share/nginx/html/
COPY services/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
