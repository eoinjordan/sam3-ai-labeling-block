---
license: apache-2.0
tags:
  - edge-impulse
  - edge-impulse-block
  - ai-labeling
  - object-detection
  - sam3
  - zero-shot
---

# SAM 3 AI labelling block for Edge Impulse

An [AI labelling block](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-ai-labeling-blocks) that labels object detection datasets in Edge Impulse Studio from text prompts. Write `hard hat (hardhat, 0.5)` and [SAM 3](https://huggingface.co/facebook/sam3) boxes every hard hat it finds in the selected images.

SAM 3 runs in the Hugging Face Space [eoinedge/sam3](https://huggingface.co/spaces/eoinedge/sam3), on ZeroGPU. The block sends each image to the Space's `/detect` endpoint and writes the boxes back to your project, so no GPU spins up in Edge Impulse and the block holds no model weights.

It is built on the same pattern as Edge Impulse's [OWL-ViT labelling block](https://github.com/edgeimpulse/zero-shot-object-detector-labeling-block), with the same prompt syntax.

## What it does

For each selected sample:

1. Fetches the image from your project (non-image samples are skipped).
2. Sends it to the Space with every phrase in the prompt, up to 8 per call.
3. Keeps boxes at or above each line's minimum confidence, clips them to the image, drops any outside the size limits, and removes overlapping boxes of the same label.
4. Combines them with the sample's existing boxes (see *Existing bounding boxes*) and writes them, with `labeled_by: sam3` and the prompt in the sample's metadata.

In **Label preview data**, Studio passes `--propose-actions`, and the boxes are proposed for you to review instead of written.

## Parameters

| Setting | Default | What it does |
|---|---|---|
| Prompt | `hard hat (hardhat, 0.5)` | One object per line: `phrase (label, min confidence)`. Without parentheses the phrase is the label and the minimum confidence is 0.5 |
| Existing bounding boxes | Keep them | **Keep them** adds only boxes that don't repeat an existing box of the same label (IoU 0.5). **Replace those in the prompt** drops existing boxes whose label is in the prompt. **Replace all** drops every existing box |
| Ignore objects smaller / larger than (%) | 0 / 100 | Box area as a share of the image |
| Non-max suppression | on, IoU 0.5 | Merges overlapping boxes of the same label |
| Hugging Face Space | `eoinedge/sam3` | Point it at your own copy for big datasets or private images |
| Hugging Face token | empty | Optional. Calls use this account's ZeroGPU quota; a PRO account has far more. Needed for a private Space |

SAM 3 works best with short noun phrases: `forklift`, `person wearing a hard hat`, `yellow school bus`.

## Push it to your Edge Impulse organisation

Custom AI labelling blocks need an Edge Impulse **Enterprise** organisation. You need the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/tools/edge-impulse-cli/cli-installation):

```bash
npm install -g edge-impulse-cli
git clone https://huggingface.co/eoinedge/sam3-ai-labeling-block
cd sam3-ai-labeling-block
edge-impulse-blocks init    # log in and choose your organisation
edge-impulse-blocks push
```

The pushed block is listed under your organisation's **Custom blocks > Transformation**; that is also where to edit it later. Then in a project: **Data acquisition > AI labeling**, choose **Bounding box labeling with SAM 3**, write the prompt, and run **Label preview data** on a few images before labelling the rest.

## Limits

- **Images leave Edge Impulse.** They are sent to the Hugging Face Space, which processes them in memory and doesn't store them. For data that must not leave your organisation, run your own copy of the Space privately, or don't use this block.
- **ZeroGPU quota.** Each caller gets a daily allowance of GPU time. A few hundred images fit in a PRO account's quota; for thousands, duplicate the Space onto paid GPU hardware and set *Hugging Face Space* to your copy. When the quota runs out the block stops, reports which samples were not labelled, and exits with an error.
- **A sleeping Space** takes a minute or two to wake on the first call.
- About 3 seconds per image with three phrases when the Space is warm (7 s for the first call), plus queueing when others are using it. A thousand images is close to an hour.

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
# the sample IDs, from Data acquisition (expand a sample to see its ID)
echo [1299267659, 1299267609] > ids.json
EI_PROJECT_API_KEY=ei_... .venv/Scripts/python label.py --prompt "hard hat (hardhat, 0.5)" --data-ids-file ids.json --nms
```

Add `--propose-actions <job id>` only when testing a preview job. The `edge-impulse-blocks` runner doesn't run AI labelling blocks, so this, or `docker run` with the same environment variable and arguments, is how to test before a push.

## Tests

```bash
.venv/Scripts/python tests/test_label.py
SPACE=eoinedge/sam3 .venv/Scripts/python tests/test_label.py   # also calls the Space
```

The tests need no Edge Impulse key: the project API is faked, but every request goes through the real `edgeimpulse_api` models. They cover the prompt syntax, both IDs file formats (a JSON array, or an object with `ids`), clipping, thresholds, size limits, NMS, the three existing-box modes, written and proposed changes, and skipped non-image samples. On 2026-10-09 the live test ran against a local copy of the Space with random weights: 10 phrases became 2 calls, with boxes in the original image's pixels. Against eoinedge/sam3 itself, COCO's two-cats photo (640x480) with `cat`, `remote control` and `couch` gave both cats, both remotes and the couch, all scoring 0.93 or more, in 3.1 s warm.

## Licences

- This block: Apache-2.0 (`LICENSE`).
- SAM 3 is Meta's model under the [SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE). The block only calls the Space; neither holds a copy of the weights in this repository.
