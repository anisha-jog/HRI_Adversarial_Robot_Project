# Gemini API Configuration
API_KEY = "AIzaSyCOA762wyugeTvRWh9cZ3Jek9LbGxn6SyQ"

# Canvas configuration
RESOLUTION = 40
PAGE_SIZE = (8.5, 11)  # in inches
CANVAS_SIZE = (int(PAGE_SIZE[1] * RESOLUTION), int(PAGE_SIZE[0] * RESOLUTION), 3)

# Axes of "Adversarial" Change
VISUAL = {
    "similar": "Very similar in shape, geometry, or style",
    "neutral": "Slightly different in shape, geometry, or style",
    "different": "Very different in shape, geometry, or style"
}
SEMANTIC = {
    "similar": "Very similar in concept or meaning",
    "neutral": "Slightly different in concept or meaning",
    "different": "Very different in concept or meaning"
}
COMPOSITIONAL = {
    "similar": "Very similar in composition, balance or layout",
    "neutral": "Slightly different in composition, balance or layout",
    "different": "Very different in composition, balance or layout"
}

# Basic "Prompt" Templates
COLLABORATIVE = {
    "visual": VISUAL["similar"],
    "semantic": SEMANTIC["similar"],
    "compositional": COMPOSITIONAL["similar"]
}
ADVERSARIAL = {
    "visual": VISUAL["neutral"],
    "semantic": SEMANTIC["neutral"],
    "compositional": COMPOSITIONAL["neutral"]
}
ANTAGONISTIC = {
    "visual": VISUAL["different"],
    "semantic": SEMANTIC["different"],
    "compositional": COMPOSITIONAL["different"]
}
CUSTOM = {
    "visual": VISUAL["similar"],
    "semantic": SEMANTIC["similar"],
    "compositional": COMPOSITIONAL["similar"]
}
CONDITIONS = {
    "collaborative": COLLABORATIVE,
    "adversarial": ADVERSARIAL,
    "antagonistic": ANTAGONISTIC,
    "custom": CUSTOM
}

# Main Gemini Prompt Template
GEMINI_PROMPT ="""
You are a creative painting agent collaborating with a user.
Given this partial drawing from the user, infer the general context that the user is trying to draw, your role is not to imitate, but to continue the human drawing in a constructive, surprising, and meaningful way.
Always explain your reasoning before generating the next stroke suggestion.
You must keep this format for the new drawing:
1. Keep the original drawing intact.
2. Overlay new drawing with red strokes on the original drawing.
3. Do not draw over the original drawing.
4. The thickness of new strokes should uniform, which is same as the thickness of the user's strokes.
5. Do not use blocks of color, only use lines.
6. The return canvas resolution should be {canvas_size}.
7. The new drawing should be feasible to draw with only lines and it is simple enough so that we can be drawn within 5 strokes.


For each attribute of the drawing, the style ranges from different, related but not identical, to similar.
For this drawing, use the following attributes in your response:
"""

SUBSEQUENT_PROMPT = """
Remeber your previous response. 
Continue to build upon the existing drawing by adding new strokes in red.

As before, follow these guidelines:
1. Keep the original drawing intact.
2. Overlay new drawing with red strokes on the original drawing, make sure to not draw over the original drawing unless necessary.
3. The thickness of new strokes should uniform, which is same as the thickness of the user's strokes.
4. Do not use blocks of color, only use lines.
5. The return canvas resolution should be {canvas_size}.
6. The new drawing should be feasible to draw with only lines and it is simple enough so that we can be drawn within 5 strokes."""

### Original prompt: 
"""
You are a creative painting agent collaborating with a user.
Given this partial drawing from the user, infer the general context
that the user is trying to draw, your role is not to imitate, but to continue the human drawing in a constructive, surprising, and meaningful way.
You can construct a new image based on the current drawing using three axes of change:
(1) Visually (unrelated shape, color, texture),
(2) Semantically (unrelated concept or meaning),
(3) Compositionally (unrelated balance or layout).
Always explain your reasoning before generating the next stroke suggestion.

You must keep this format for the new drawing:
1. Keep the original drawing intact.
2. Overlay new drawing with red strokes on the original drawing.
3. The thickness of new strokes should uniform, which is same as the thickness of the user's strokes.
4. The return canvas resolution should be {canvas_size}.
5. The new drawing should be feasible to draw with lines and it is simple enough so that we can be drawn within 5 strokes.
""" 