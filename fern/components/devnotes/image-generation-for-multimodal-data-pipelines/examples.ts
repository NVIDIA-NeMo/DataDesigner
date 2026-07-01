/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ImageExampleGalleryItem } from "../../ImageExampleGallery";

export const productImageVariationExamples: ImageExampleGalleryItem[] = [
    {
      title: "Example 1: adaptive zip-front jacket · adaptive-fashion catalog image focused on ease of wear",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/product-image-variations/01.jpg",
      alt: "Before and after adult apparel image variation",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["apparel item", "adaptive zip-front jacket"],
            ["base colorway", "brushed silver"],
            ["base view", "front-facing standing full-body catalog photo with one synthetic adult model"],
            ["base model profile", "young adult South Asian model with a plus-size build"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["variation goal", "adaptive-fashion catalog image focused on ease of wear"],
            ["model age group", "middle-aged adult model"],
            ["model ethnicity", "White or European model"],
            ["body type", "broad-shouldered build"],
            ["accessibility context", "seated model without visible mobility aids"],
            ["composition", "single seated full-body pose showing garment fit with the whole body visible"],
            ["lighting", "softbox studio lighting"],
          ],
        },
      ],
    },
    {
      title: "Example 2: quilted puffer vest · editorial fashion image with soft premium lighting",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/product-image-variations/02.png",
      alt: "Before and after adult apparel image variation",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["apparel item", "quilted puffer vest"],
            ["base colorway", "denim blue"],
            ["base view", "walking-pose full-body catalog photo with one synthetic adult model"],
            ["base model profile", "young adult Indigenous model with a broad-shouldered build"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["variation goal", "editorial fashion image with soft premium lighting"],
            ["edit scene delta", "move from a neutral studio catalog reference into a smart-casual workplace corridor scene"],
            ["model age group", "senior adult model"],
            ["model ethnicity", "Pacific Islander model"],
            ["body type", "petite build"],
            ["accessibility context", "model with no specific accessibility cue"],
            ["styling context", "minimal geometric studio set"],
            ["composition", "side-angle full-body pose with clear garment silhouette"],
            ["lighting", "natural window light"],
          ],
        },
      ],
    },
    {
      title: "Example 3: ribbed knit cardigan · lifestyle lookbook image in an everyday urban setting",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/product-image-variations/03.png",
      alt: "Before and after adult apparel image variation",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["apparel item", "ribbed knit cardigan"],
            ["base colorway", "terracotta"],
            ["base view", "three-quarter standing full-body catalog photo with one synthetic adult model"],
            ["base model profile", "older White model with a curvy build"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["variation goal", "lifestyle lookbook image in an everyday urban setting"],
            ["edit scene delta", "move from a neutral studio catalog reference into a premium editorial studio set with draped fabric"],
            ["model age group", "middle-aged adult model"],
            ["model ethnicity", "South Asian model"],
            ["body type", "tall build"],
            ["accessibility context", "model with no specific accessibility cue"],
            ["styling context", "weekend park setting"],
            ["composition", "single walking full-body pose with natural garment movement"],
            ["lighting", "softbox studio lighting"],
          ],
        },
      ],
    },
    {
      title: "Example 4: water-resistant rain jacket · editorial fashion image with soft premium lighting",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/product-image-variations/04.jpg",
      alt: "Before and after adult apparel image variation",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["apparel item", "water-resistant rain jacket"],
            ["base colorway", "sunflower yellow"],
            ["base view", "walking-pose full-body catalog photo with one synthetic adult model"],
            ["base model profile", "young adult South Asian model with a plus-size build"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["variation goal", "editorial fashion image with soft premium lighting"],
            ["model age group", "young adult model"],
            ["model ethnicity", "Black or African diaspora model"],
            ["body type", "broad-shouldered build"],
            ["accessibility context", "model leaning lightly against a studio block"],
            ["composition", "front-facing full-body catalog pose with the entire person visible"],
            ["lighting", "soft overcast outdoor light"],
          ],
        },
      ],
    },
    {
      title: "Example 5: wide-leg linen trousers · editorial fashion image with soft premium lighting",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/product-image-variations/05.png",
      alt: "Before and after adult apparel image variation",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["apparel item", "wide-leg linen trousers"],
            ["base colorway", "charcoal gray"],
            ["base view", "front-facing standing full-body catalog photo with one synthetic adult model"],
            ["base model profile", "young adult multiracial model with a slender build"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["variation goal", "editorial fashion image with soft premium lighting"],
            ["edit scene delta", "move from a neutral studio catalog reference into a warm home entryway lookbook scene"],
            ["model age group", "senior adult model"],
            ["model ethnicity", "Black or African diaspora model"],
            ["body type", "tall build"],
            ["accessibility context", "seated model without visible mobility aids"],
            ["styling context", "minimal geometric studio set"],
            ["composition", "single seated full-body pose showing garment fit with the whole body visible"],
            ["lighting", "natural window light"],
          ],
        },
      ],
    },
  ];
export const trafficScenarioExamples: ImageExampleGalleryItem[] = [
    {
      title: "Example 1: rural two-lane country road · pedestrian jaywalking mid-block",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/traffic-scenarios/01.jpg",
      alt: "Autonomous vehicle ego-camera traffic scene",
      controls: [
        ["geographic region", "US - sprawling suburban (Los Angeles-style)"],
        ["road type", "rural two-lane country road"],
        ["weather", "dense fog - visibility under 50ft"],
        ["time of day", "mid-morning - diffuse daylight through fog"],
        ["traffic density", "light - 3-5 vehicles visible"],
        ["vehicle mix", "includes delivery vans/box trucks"],
        ["scenario element", "pedestrian jaywalking mid-block"],
        ["road surface", "potholes and road damage visible"],
        ["traffic control", "construction signage and cones"],
      ],
    },
    {
      title: "Example 2: urban street with mixed retail/residential · cyclist making left turn",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/traffic-scenarios/02.jpg",
      alt: "Autonomous vehicle ego-camera traffic scene",
      controls: [
        ["geographic region", "US - rural Midwest"],
        ["road type", "urban street with mixed retail/residential"],
        ["weather", "dense fog - visibility under 50ft"],
        ["time of day", "dusk - low-visibility fog"],
        ["traffic density", "stop-and-go - bumper-to-bumper"],
        ["vehicle mix", "sedans and compact cars"],
        ["scenario element", "cyclist making left turn"],
        ["road surface", "recent patching - uneven surface"],
        ["traffic control", "construction signage and cones"],
      ],
    },
    {
      title: "Example 3: intersection - 4-way with traffic lights · animal darting across road",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/traffic-scenarios/03.jpg",
      alt: "Autonomous vehicle ego-camera traffic scene",
      controls: [
        ["geographic region", "Asia - mixed traffic (India/Thailand)"],
        ["road type", "intersection - 4-way with traffic lights"],
        ["weather", "dust storm - desert conditions"],
        ["time of day", "night - moderately lit urban"],
        ["traffic density", "sparse - 1-2 vehicles in distance"],
        ["vehicle mix", "includes large trucks/semi-trailers"],
        ["scenario element", "animal (coyote/fox) darting across road"],
        ["road surface", "dry asphalt - good condition"],
        ["traffic control", "traffic light - green"],
      ],
    },
    {
      title: "Example 4: bridge - concrete overpass · traffic barrel knocked into lane",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/traffic-scenarios/04.jpg",
      alt: "Autonomous vehicle ego-camera traffic scene",
      controls: [
        ["geographic region", "US - rural Midwest"],
        ["road type", "bridge - concrete overpass"],
        ["weather", "moderate rain - active wipers"],
        ["time of day", "early morning - golden-gray light"],
        ["traffic density", "stop-and-go - bumper-to-bumper"],
        ["vehicle mix", "sedans and compact cars"],
        ["scenario element", "traffic barrel knocked into lane"],
        ["road surface", "wet with puddles"],
        ["traffic control", "lane-closure arrow board and construction barrels"],
      ],
    },
    {
      title: "Example 5: rural motorway · disabled vehicle on shoulder",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/traffic-scenarios/05.jpg",
      alt: "Autonomous vehicle ego-camera traffic scene",
      controls: [
        ["geographic region", "Europe - orderly infrastructure (German autobahn)"],
        ["road type", "rural motorway / autobahn-style highway"],
        ["weather", "clear dusk"],
        ["time of day", "dusk - post-sunset twilight"],
        ["traffic density", "sparse - 1-2 vehicles in distance"],
        ["vehicle mix", "includes bus/coach"],
        ["scenario element", "disabled vehicle on shoulder with warning triangle"],
        ["road surface", "dry pavement with clear lane markings"],
        ["traffic control", "right-lane merge advisory sign"],
      ],
    },
  ];

export const humanoidRobotSceneExamples: ImageExampleGalleryItem[] = [
    {
      title: "Example 1: hospital supply room · power cable crossing the walking path",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/humanoid-robot-scene-understanding/01.jpg",
      alt: "Egocentric humanoid robot scene-understanding image",
      controls: [
        ["environment", "hospital supply room with carts, bins, and sealed supplies"],
        ["robot viewpoint", "chest-mounted camera with both robot hands barely visible at the bottom edge"],
        ["task goal", "identify which objects are reachable from the current pose"],
        ["object set", "meal tray, sealed supplies, clipboard, and rolling cart"],
        ["scene state", "container open with mixed contents visible"],
        ["safety condition", "power cable crossing the walking path"],
        ["human presence", "adult caregiver standing in the background with face turned away"],
      ],
    },
    {
      title: "Example 2: mock apartment living room · close manipulation view",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/humanoid-robot-scene-understanding/02.jpg",
      alt: "Egocentric humanoid robot scene-understanding image",
      controls: [
        ["environment", "mock apartment living room arranged for assistive robotics"],
        ["robot viewpoint", "close manipulation view with one robot hand near the target object"],
        ["task goal", "verify that a cleanup task is complete"],
        ["object set", "pipette rack, beaker, nitrile gloves, and small screwdriver"],
        ["scene state", "container open with mixed contents visible"],
        ["safety condition", "no visible hazard"],
        ["human presence", "adult office worker's arm visible near the handoff area"],
      ],
    },
    {
      title: "Example 3: office break room · inspect a spill or obstacle",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/humanoid-robot-scene-understanding/03.jpg",
      alt: "Egocentric humanoid robot scene-understanding image",
      controls: [
        ["environment", "office break room with appliances, tableware, and waste bins"],
        ["robot viewpoint", "chest-mounted camera with both robot hands barely visible at the bottom edge"],
        ["task goal", "inspect a spill or obstacle before moving closer"],
        ["object set", "mug, kettle, sponge, dish towel, and cereal bowl"],
        ["scene state", "task area partly blocked by a chair or cart"],
        ["safety condition", "power cable crossing the walking path"],
        ["lighting", "high-contrast backlighting from a nearby window"],
      ],
    },
    {
      title: "Example 4: low crouched inspection · power cable crossing the walking path",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/humanoid-robot-scene-understanding/04.jpg",
      alt: "Egocentric humanoid robot scene-understanding image",
      controls: [
        ["environment", "mock apartment living room arranged for assistive robotics"],
        ["robot viewpoint", "low crouched inspection angle looking under a table or cart"],
        ["task goal", "decide whether fragile items are too close to an edge"],
        ["object set", "barcode scanner, tote, tape dispenser, folded shirt, and box cutter"],
        ["scene state", "container open with mixed contents visible"],
        ["safety condition", "power cable crossing the walking path"],
        ["lighting", "dim hallway light with localized task lamp"],
      ],
    },
    {
      title: "Example 5: retail stockroom · fragile item near the table edge",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/humanoid-robot-scene-understanding/05.jpg",
      alt: "Egocentric humanoid robot scene-understanding image",
      controls: [
        ["environment", "retail stockroom with shelves, totes, and handheld items"],
        ["robot viewpoint", "head-mounted camera at standing adult height"],
        ["task goal", "identify which objects are reachable from the current pose"],
        ["object set", "pliers, hex keys, small bolts, tape measure, and plastic bins"],
        ["scene state", "fragile item near the table edge"],
        ["safety condition", "no visible hazard"],
        ["human presence", "no person visible"],
      ],
    },
  ];

export const airportSecurityScanExamples: ImageExampleGalleryItem[] = [
    {
      title: "Example 1: messenger bag · clutter and overlap preventing confident clearance",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/airport-security-scans/01.jpg",
      alt: "Synthetic defensive airport baggage screening image",
      controls: [
        ["scanner style", "dual-energy X-ray baggage scan with pseudo-color material mapping"],
        ["bag type", "messenger bag"],
        ["bag density", "very dense packing with cluttered object boundaries"],
        ["benign contents", "business travel items, documents, laptop, and power adapters"],
        ["material mix", "many small overlapping metal and plastic objects"],
        ["threat type", "clutter and overlapping objects preventing confident clearance"],
      ],
    },
    {
      title: "Example 2: hard-shell suitcase · oversized liquid-container-like region",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/airport-security-scans/02.jpg",
      alt: "Synthetic defensive airport baggage screening image",
      controls: [
        ["scanner style", "side-view checked-bag screening image"],
        ["bag type", "hard-shell suitcase"],
        ["bag density", "sparse packing with many empty regions"],
        ["benign contents", "camera body, lenses, batteries, cables, and clothing"],
        ["material mix", "mostly fabric and plastic with a few small metal objects"],
        ["threat type", "oversized liquid-container-like region requiring secondary review"],
      ],
    },
    {
      title: "Example 3: soft backpack · unknown dense object",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/airport-security-scans/03.jpg",
      alt: "Synthetic defensive airport baggage screening image",
      controls: [
        ["scanner style", "computed tomography baggage scan slice rendered as pseudo-color X-ray"],
        ["bag type", "soft backpack"],
        ["bag density", "dense packing with overlapping objects"],
        ["benign contents", "clothing, shoes, toiletries, and paperback books"],
        ["material mix", "many small overlapping metal and plastic objects"],
        ["threat type", "unknown dense object requiring secondary review"],
      ],
    },
    {
      title: "Example 4: duffel bag · dense electronics cluster",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/airport-security-scans/04.jpg",
      alt: "Synthetic defensive airport baggage screening image",
      controls: [
        ["scanner style", "top-down carry-on baggage screening view"],
        ["bag type", "duffel bag"],
        ["bag density", "very dense packing with cluttered object boundaries"],
        ["benign contents", "laptop, chargers, headphones, notebooks, and snacks"],
        ["material mix", "electronics-heavy bag with cables and batteries"],
        ["threat type", "dense electronics cluster requiring secondary review"],
      ],
    },
    {
      title: "Example 5: soft backpack · oversized liquid-container-like region",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/airport-security-scans/05.jpg",
      alt: "Synthetic defensive airport baggage screening image",
      controls: [
        ["scanner style", "dual-energy X-ray baggage scan with pseudo-color material mapping"],
        ["bag type", "soft backpack"],
        ["bag density", "very dense packing with cluttered object boundaries"],
        ["benign contents", "children's toys, folded clothing, tablet, and water bottle"],
        ["material mix", "mixed organic, plastic, and metal materials"],
        ["threat type", "oversized liquid-container-like region requiring secondary review"],
      ],
    },
  ];

export const medicalExtremityXrayExamples: ImageExampleGalleryItem[] = [
    {
      title: "Example 1: left hip · osteoarthritis with joint space narrowing and osteophytes",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/medical-extremity-xrays/01.jpg",
      alt: "Synthetic research-only extremity X-ray style image",
      controls: [
        ["anatomical region", "left hip"],
        ["xray view", "oblique external rotation"],
        ["primary finding", "osteoarthritis with joint space narrowing and osteophytes"],
        ["secondary findings", "old healed fracture with callus formation"],
        ["exposure quality", "high kVp technique with better soft tissue visualization"],
        ["image quality", "limited by patient body habitus"],
      ],
    },
    {
      title: "Example 2: left hand and fingers · boxer's fracture of the fifth metacarpal neck",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/medical-extremity-xrays/02.jpg",
      alt: "Synthetic research-only extremity X-ray style image",
      controls: [
        ["anatomical region", "left hand and fingers"],
        ["xray view", "anteroposterior (AP)"],
        ["primary finding", "boxer's fracture of the fifth metacarpal neck"],
        ["secondary findings", "soft tissue calcifications"],
        ["exposure quality", "optimal exposure with clear cortical and trabecular detail"],
        ["image quality", "fair with mild motion artifact"],
      ],
    },
    {
      title: "Example 3: right shoulder · osteomyelitis with cortical destruction",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/medical-extremity-xrays/03.jpg",
      alt: "Synthetic research-only extremity X-ray style image",
      controls: [
        ["anatomical region", "right shoulder"],
        ["xray view", "lateral"],
        ["primary finding", "osteomyelitis with cortical destruction"],
        ["secondary findings", "osteopenia"],
        ["exposure quality", "underexposed with cortical margins poorly defined"],
        ["image quality", "fair with mild noise or graininess"],
      ],
    },
    {
      title: "Example 4: left tibia and fibula · displaced fracture through the imaged bone",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/medical-extremity-xrays/04.jpg",
      alt: "Synthetic research-only extremity X-ray style image",
      controls: [
        ["anatomical region", "left tibia and fibula"],
        ["xray view", "stress view"],
        ["primary finding", "displaced fracture through the imaged bone"],
        ["secondary findings", "none"],
        ["exposure quality", "low kVp technique with high bone contrast"],
        ["image quality", "fair with cast or splint partially obscuring detail"],
      ],
    },
    {
      title: "Example 5: right knee · osteoarthritis with joint space narrowing and osteophytes",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/medical-extremity-xrays/05.jpg",
      alt: "Synthetic research-only extremity X-ray style image",
      controls: [
        ["anatomical region", "right knee"],
        ["xray view", "anteroposterior (AP)"],
        ["primary finding", "osteoarthritis with joint space narrowing and osteophytes"],
        ["secondary findings", "soft tissue calcifications"],
        ["exposure quality", "underexposed with cortical margins poorly defined"],
        ["image quality", "fair with cast or splint partially obscuring detail"],
      ],
    },
  ];

export const agricultureCropExamples: ImageExampleGalleryItem[] = [
    {
      title: "Example 1: soybean · insect feeding damage as a disease confounder",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/crop-disease-detection-images/01.jpg",
      alt: "Synthetic crop disease detection image",
      controls: [
        ["crop type", "soybean"],
        ["growth stage", "flowering"],
        ["viewpoint", "top-down drone crop-row view"],
        ["disease or condition", "insect feeding damage as a disease confounder"],
        ["severity", "high severity affecting large field sections"],
        ["field condition", "mulched bed system"],
        ["weather lighting", "hazy smoky sky"],
      ],
    },
    {
      title: "Example 2: grape vineyard · early blight with concentric brown leaf spots",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/crop-disease-detection-images/02.jpg",
      alt: "Synthetic crop disease detection image",
      controls: [
        ["crop type", "grape vineyard"],
        ["growth stage", "flowering"],
        ["viewpoint", "greenhouse bench view"],
        ["disease or condition", "early blight with concentric brown leaf spots"],
        ["severity", "low severity affecting isolated plants"],
        ["field condition", "visible irrigation lines"],
        ["weather lighting", "golden hour light"],
      ],
    },
    {
      title: "Example 3: potato · late blight with irregular dark lesions",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/crop-disease-detection-images/03.jpg",
      alt: "Synthetic crop disease detection image",
      controls: [
        ["crop type", "potato"],
        ["growth stage", "flowering"],
        ["viewpoint", "orchard row view"],
        ["disease or condition", "late blight with irregular dark lesions"],
        ["severity", "low severity affecting isolated plants"],
        ["field condition", "dry cracked soil"],
        ["weather lighting", "golden hour light"],
      ],
    },
    {
      title: "Example 4: grape vineyard · downy mildew patches on leaf undersides",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/crop-disease-detection-images/04.jpg",
      alt: "Synthetic crop disease detection image",
      controls: [
        ["crop type", "grape vineyard"],
        ["growth stage", "grain fill"],
        ["viewpoint", "close-up leaf-level scouting photo"],
        ["disease or condition", "downy mildew patches on leaf undersides"],
        ["severity", "low severity affecting isolated plants"],
        ["field condition", "patchy emergence"],
        ["weather lighting", "golden hour light"],
      ],
    },
    {
      title: "Example 5: tomato · insect feeding damage as a disease confounder",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/crop-disease-detection-images/05.jpg",
      alt: "Synthetic crop disease detection image",
      controls: [
        ["crop type", "tomato"],
        ["growth stage", "fruiting"],
        ["viewpoint", "orchard row view"],
        ["disease or condition", "insect feeding damage as a disease confounder"],
        ["severity", "high severity affecting large field sections"],
        ["field condition", "uniform crop stand"],
        ["weather lighting", "greenhouse diffuse lighting"],
      ],
    },
  ];

export const droneAerialInspectionExamples: ImageExampleGalleryItem[] = [
    {
      title: "Example 1: utility pipeline corridor · erosion or washout near an edge",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/drone-aerial-inspection/01.jpg",
      alt: "Synthetic drone aerial inspection image",
      controls: [
        ["site type", "utility pipeline corridor"],
        ["inspection target", "standing water"],
        ["altitude", "very low drone pass, about 10 meters above the target"],
        ["camera angle", "overview frame with the target centered"],
        ["defect or event", "erosion or washout near an edge"],
        ["severity", "none"],
        ["occlusion", "partially occluded by shadows"],
      ],
    },
    {
      title: "Example 2: construction site · vegetation growth obscuring part of the asset",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/drone-aerial-inspection/02.jpg",
      alt: "Synthetic drone aerial inspection image",
      controls: [
        ["site type", "construction site"],
        ["inspection target", "roof covering condition"],
        ["altitude", "very low drone pass, about 10 meters above the target"],
        ["camera angle", "oblique 45-degree inspection angle"],
        ["defect or event", "vegetation growth obscuring part of the asset"],
        ["severity", "minor and easy to miss"],
        ["occlusion", "partially occluded by temporary equipment"],
      ],
    },
    {
      title: "Example 3: commercial flat roof · moderate staining or water pooling",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/drone-aerial-inspection/03.jpg",
      alt: "Synthetic drone aerial inspection image",
      controls: [
        ["site type", "commercial flat roof"],
        ["inspection target", "construction progress milestone"],
        ["altitude", "very low drone pass, about 10 meters above the target"],
        ["camera angle", "close detail view with wide-angle lens"],
        ["defect or event", "moderate staining or water pooling"],
        ["severity", "severe and clearly visible"],
        ["occlusion", "partially occluded by temporary equipment"],
      ],
    },
    {
      title: "Example 4: roadway and drainage culvert · surface discoloration",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/drone-aerial-inspection/04.jpg",
      alt: "Synthetic drone aerial inspection image",
      controls: [
        ["site type", "roadway and drainage culvert"],
        ["inspection target", "debris blocking access"],
        ["altitude", "low drone pass, about 25 meters above the target"],
        ["camera angle", "close detail view with wide-angle lens"],
        ["defect or event", "surface discoloration that may be benign"],
        ["severity", "moderate and localized"],
        ["occlusion", "partially occluded by temporary equipment"],
      ],
    },
    {
      title: "Example 5: bridge deck and support structure · vegetation obscuring part of the asset",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/drone-aerial-inspection/05.jpg",
      alt: "Synthetic drone aerial inspection image",
      controls: [
        ["site type", "bridge deck and support structure"],
        ["inspection target", "surface cracking"],
        ["altitude", "medium drone pass, about 60 meters above the target"],
        ["camera angle", "overview frame with the target centered"],
        ["defect or event", "vegetation growth obscuring part of the asset"],
        ["severity", "severe and clearly visible"],
        ["occlusion", "partially occluded by tree branches"],
      ],
    },
  ];

export const funnyPetImageEditExamples: ImageExampleGalleryItem[] = [
    {
      title: "Example 1: gray tabby cat · very serious chef inspecting a tiny bowl",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/funny-pet-image-edits/01.png",
      alt: "Before and after funny pet image edit",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["pet type", "cat"],
            ["pet breed", "gray tabby"],
            ["pet age", "adult cat, 4 to 10 years old"],
            ["base activity", "posing beside a cardboard box"],
            ["base setting", "tidy kitchen corner"],
            ["pet expression", "proud little smirk"],
            ["base photo style", "warm editorial pet portrait"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["comedy goal", "stage the pet as a very serious chef inspecting a tiny bowl"],
            ["funny prop", "miniature necktie"],
            ["scene escalation", "add a neatly arranged set of miniature props around the pet"],
            ["humor style", "cozy wholesome comedy"],
          ],
        },
      ],
    },
    {
      title: "Example 2: tiny long-haired dog · orchestra conductor for squeaky toys",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/funny-pet-image-edits/02.png",
      alt: "Before and after funny pet image edit",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["pet type", "dog"],
            ["pet breed", "Shih Tzu"],
            ["pet age", "adult dog, 4 to 7 years old"],
            ["base activity", "balanced calmly beside a pile of toys"],
            ["base setting", "laundry room with folded towels"],
            ["pet expression", "sleepy but determined expression"],
            ["base photo style", "slightly low-angle comedic portrait"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["comedy goal", "stage the pet as a tiny orchestra conductor for squeaky toys"],
            ["funny prop", "tiny blanket cape"],
            ["scene escalation", "add a neatly arranged set of miniature props around the pet"],
            ["humor style", "overly dramatic tiny-professional energy"],
          ],
        },
      ],
    },
    {
      title: "Example 3: scruffy terrier · blanket-cape superhero with plush audience",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/funny-pet-image-edits/03.png",
      alt: "Before and after funny pet image edit",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["pet type", "dog"],
            ["pet breed", "mixed-breed terrier"],
            ["pet age", "young adult dog, 1 to 4 years old"],
            ["base activity", "looking directly at the camera with dramatic seriousness"],
            ["base setting", "soft studio backdrop"],
            ["pet expression", "proud little smirk"],
            ["base photo style", "clean studio portrait with gentle shadows"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["comedy goal", "stage the pet as a blanket-cape superhero in a cozy room"],
            ["funny prop", "blank tiny clipboard with no writing"],
            ["scene escalation", "add an audience of plush toys in the background"],
            ["humor style", "gentle visual slapstick without distress"],
          ],
        },
      ],
    },
    {
      title: "Example 4: wide-eyed corgi · cardboard-spaceship pilot",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/funny-pet-image-edits/04.png",
      alt: "Before and after funny pet image edit",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["pet type", "dog"],
            ["pet breed", "Pembroke Welsh corgi"],
            ["pet age", "young adult dog, 1 to 4 years old"],
            ["base activity", "sitting proudly at a small table"],
            ["base setting", "soft studio backdrop"],
            ["pet expression", "wide-eyed confused expression"],
            ["base photo style", "natural phone photo with soft daylight"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["comedy goal", "stage the pet as a cardboard-spaceship pilot with abstract controls"],
            ["funny prop", "miniature trophy with no writing"],
            ["scene escalation", "add a whimsical but tidy tabletop set"],
            ["humor style", "deadpan absurdity"],
          ],
        },
      ],
    },
    {
      title: "Example 5: scruffy terrier · toy stage performer under a tiny spotlight",
      src: "/assets/image-generation-for-multimodal-data-pipelines/examples/funny-pet-image-edits/05.png",
      alt: "Before and after funny pet image edit",
      controlGroups: [
        {
          label: "Initial image controls",
          controls: [
            ["pet type", "dog"],
            ["pet breed", "mixed-breed terrier"],
            ["pet age", "adult dog, 4 to 7 years old"],
            ["base activity", "sitting proudly at a small table"],
            ["base setting", "soft studio backdrop"],
            ["pet expression", "deeply serious expression"],
            ["base photo style", "slightly low-angle comedic portrait"],
          ],
        },
        {
          label: "Edit conditions",
          controls: [
            ["comedy goal", "stage the pet as a toy stage performer under a tiny spotlight"],
            ["funny prop", "miniature trophy with no writing"],
            ["scene escalation", "add a playful spotlight and dramatic shadows"],
            ["humor style", "cozy wholesome comedy"],
          ],
        },
      ],
    },
  ];
