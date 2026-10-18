# Front React

Interface de recherche : un champ de texte, un bouton « Search », et la liste des résultats avec leur cluster et leur score de pertinence.

## Lancement

En développement, depuis ce dossier :

```powershell
npm install
npm run dev
```

Puis ouvrir http://localhost:5173.

Avec Docker, depuis `iasearch` :

```powershell
docker compose up -d --build front
```

Puis ouvrir http://localhost:3000.

## Fichiers principaux

- `src/App.tsx` : champ de recherche, appel à `POST /search`, affichage des résultats.
- `src/App.css` : mise en forme.