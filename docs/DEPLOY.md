# Deploying

## The public demo: Streamlit Community Cloud (free, no card)

The live demo runs on Streamlit Community Cloud from this repository's `streamlit_app.py`: the API runs inside the
Streamlit process (a background thread on 127.0.0.1 only) because Community Cloud hosts one Streamlit app and no
Docker. The reasoning agent serves the Gemma answers recorded on a GPU (`backend/data/replay/answers.jsonl`) for the
demo presets, labelled as recorded. `requirements.txt` pins the numeric libraries to the API image's versions:
recorded answers are matched by exact numbers. Measured in a fresh Python 3.12 install like Community Cloud's:
137/137 answers served from the recordings.

1. Sign in at https://share.streamlit.io with GitHub.
2. *Create app* → *Deploy a public app from GitHub*: repository `a1if/ecg-multi-agent-triage`, branch `main`,
   main file `streamlit_app.py`; *Advanced settings* → Python **3.12**. Deploy (the first build takes a few minutes).
3. The app sleeps after a period without visitors; the next visitor wakes it with one click.

Hosting attempts that did not fit, for the record: Google Cloud Run (needs a billing account; this card required a
refundable prepayment), Hugging Face Docker Spaces (free CPU hardware now needs a PRO subscription for Docker and
Gradio Spaces). The Hugging Face Space is still buildable (`deploy/hf-space`, `backend/scripts/build_hf_space.py`).

# Google Cloud (defined as code, not running)

The public demo runs on **CPU only**: the API and frontend on Cloud Run, serving **real Gemma answers recorded on a
GPU** for fixed presets (each bundled record, first 5 minutes) and labelling them as recorded. The GPU service exists
in the Terraform (`enable_gpu = true`) but is off by default, because the GPU is almost the whole cost.

```
browser ─> ecg-frontend (public) ─ID token─> ecg-api (private) ─ID token─> ecg-inference (L4 GPU, only if enabled)
GitHub Actions ─OIDC (no keys)─> deployer service account ─> Artifact Registry + Cloud Run
```

## Cost (CPU-only demo)

| Part | Expected cost |
|---|---|
| Frontend (Cloud Run, request-based billing) | free tier covers demo traffic |
| API (Cloud Run, instance-based billing so the agents can run after answering 202) | a few cents per demo session; scales to zero |
| Artifact Registry (old images deleted automatically, newest 3 kept) | well under $1 / month |
| Inference with an L4 GPU (`enable_gpu = true`) | billed for every second an instance is alive, including idle time before it scales down; check the current rate at https://cloud.google.com/run/pricing first. Needs a paid (upgraded) billing account |

## 1. Account and project (you, in the browser)

1. Go to https://console.cloud.google.com and sign up. The free trial gives $300 of credit for 90 days, and
   **nothing is charged unless you later upgrade** to a paid account yourself.
2. Create a project (top bar → project picker → *New project*), e.g. `ecg-triage-demo`. Note its **project ID**
   (it may have a numeric suffix).
3. **Budget alert:** *Billing → Budgets & alerts → Create budget*: scope this project, amount **$5**, alerts at 50%,
   90% and 100% (emails go to you). A budget *warns*; it does not stop spending.

## 2. Sign in from this machine (you, one time)

In PowerShell (new window, so the newly installed `gcloud` is on the PATH):

```powershell
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Both logins open the browser. The second one is what Terraform uses.

## 3. Create the infrastructure (Terraform)

```powershell
cd C:\Users\alift\ecg-triage-agent\infra\terraform
terraform init
terraform apply -var project_id=YOUR_PROJECT_ID
```

`apply` prints the plan and asks `yes` before changing anything. It creates the services with a placeholder
image ("Hello" page); the deploy workflow replaces it with the real images.

## 4. Let GitHub deploy (repository variables, not secrets)

```powershell
terraform output github_variables
```

Set each as a repository variable (they are identifiers, not credentials):

```powershell
gh variable set GCP_PROJECT_ID --body "..."
gh variable set GCP_REGION --body "us-central1"
gh variable set GCP_WIF_PROVIDER --body "projects/.../providers/github-oidc"
gh variable set GCP_DEPLOYER_SA --body "ecg-deployer@....iam.gserviceaccount.com"
```

Then run the **deploy** workflow (Actions → deploy → *Run workflow*), or push to `main`: after CI passes, the
tested images are copied to Artifact Registry and rolled out, and the job summary shows the public URL.

## 5. Turning the GPU on (optional, paid account)

1. Upgrade the billing account (Billing → *Upgrade*). Remaining trial credit is used first.
2. `terraform apply -var project_id=... -var enable_gpu=true`: creates the bucket for Gemma's weights and the
   private GPU service; the API switches to `RECEIVER=http` and turns the demo presets off.
3. Copy Gemma's weights into the bucket (`gcloud storage cp -r ~/.cache/huggingface/hub gs://PROJECT-ecg-models/`),
   push the inference image, and deploy it.
4. Turn it off again with `-var enable_gpu=false`, or set its maximum instances to 0.

## Tearing everything down

```powershell
terraform destroy -var project_id=YOUR_PROJECT_ID
```

Then delete the project in the console if you want nothing left at all.
