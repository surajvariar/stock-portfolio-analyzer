<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Frontend and stock API

- Configure stock API requests through `VITE_STOCK_API_URL` (default `http://localhost:8000` for local development); a 401 from Kite holdings sync redirects to that API's `/login` endpoint so expired sessions can be renewed.
