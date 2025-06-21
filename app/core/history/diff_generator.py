import difflib
from typing import List, Dict, Any, Tuple, Optional
import re
import json
from datetime import datetime
import hashlib

def generate_document_diff(text1: str, text2: str, context_lines: int = 3) -> Dict[str, Any]:
    """Generate a comprehensive visual diff between two document texts."""

    # Split texts into lines
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Generate unified diff
    diff = difflib.unified_diff(
        lines1,
        lines2,
        lineterm='',
        n=context_lines
    )

    # Format diff for display
    diff_lines = list(diff)

    # Skip the first two lines (header)
    if len(diff_lines) > 2:
        diff_lines = diff_lines[2:]

    # Generate side-by-side diff
    side_by_side = generate_side_by_side_diff(lines1, lines2)

    # Generate inline diff with word-level highlighting
    inline_diff = generate_inline_diff(text1, text2)

    # Calculate detailed statistics
    stats = calculate_diff_statistics(lines1, lines2, diff_lines)

    # Format for HTML display
    html_diff = format_html_diff(diff_lines)

    return {
        "diff_text": '\n'.join(diff_lines),
        "html_diff": html_diff,
        "side_by_side": side_by_side,
        "inline_diff": inline_diff,
        "stats": stats,
        "similarity_score": calculate_text_similarity(text1, text2)
    }


def generate_side_by_side_diff(lines1: List[str], lines2: List[str]) -> Dict[str, Any]:
    """Generate side-by-side diff view."""

    differ = difflib.SequenceMatcher(None, lines1, lines2)
    side_by_side = {
        "left_lines": [],
        "right_lines": [],
        "line_pairs": []
    }

    for tag, i1, i2, j1, j2 in differ.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2):
                side_by_side["line_pairs"].append({
                    "left": {"line_num": i + 1, "text": lines1[i], "type": "equal"},
                    "right": {"line_num": j1 + (i - i1) + 1, "text": lines2[j1 + (i - i1)], "type": "equal"}
                })
        elif tag == 'delete':
            for i in range(i1, i2):
                side_by_side["line_pairs"].append({
                    "left": {"line_num": i + 1, "text": lines1[i], "type": "delete"},
                    "right": {"line_num": None, "text": "", "type": "empty"}
                })
        elif tag == 'insert':
            for j in range(j1, j2):
                side_by_side["line_pairs"].append({
                    "left": {"line_num": None, "text": "", "type": "empty"},
                    "right": {"line_num": j + 1, "text": lines2[j], "type": "insert"}
                })
        elif tag == 'replace':
            # Handle replacements by pairing lines when possible
            max_lines = max(i2 - i1, j2 - j1)
            for k in range(max_lines):
                left_line = lines1[i1 + k] if k < (i2 - i1) else ""
                right_line = lines2[j1 + k] if k < (j2 - j1) else ""
                left_num = (i1 + k + 1) if k < (i2 - i1) else None
                right_num = (j1 + k + 1) if k < (j2 - j1) else None

                side_by_side["line_pairs"].append({
                    "left": {"line_num": left_num, "text": left_line, "type": "replace" if left_line else "empty"},
                    "right": {"line_num": right_num, "text": right_line, "type": "replace" if right_line else "empty"}
                })

    return side_by_side


def generate_inline_diff(text1: str, text2: str) -> Dict[str, Any]:
    """Generate inline diff with word-level highlighting."""

    # Split into words for more granular comparison
    words1 = re.findall(r'\S+|\s+', text1)
    words2 = re.findall(r'\S+|\s+', text2)

    differ = difflib.SequenceMatcher(None, words1, words2)
    inline_result = []

    for tag, i1, i2, j1, j2 in differ.get_opcodes():
        if tag == 'equal':
            inline_result.append({
                "type": "equal",
                "text": ''.join(words1[i1:i2])
            })
        elif tag == 'delete':
            inline_result.append({
                "type": "delete",
                "text": ''.join(words1[i1:i2])
            })
        elif tag == 'insert':
            inline_result.append({
                "type": "insert",
                "text": ''.join(words2[j1:j2])
            })
        elif tag == 'replace':
            inline_result.append({
                "type": "delete",
                "text": ''.join(words1[i1:i2])
            })
            inline_result.append({
                "type": "insert",
                "text": ''.join(words2[j1:j2])
            })

    return {
        "segments": inline_result,
        "html": format_inline_html(inline_result)
    }


def calculate_diff_statistics(lines1: List[str], lines2: List[str], diff_lines: List[str]) -> Dict[str, Any]:
    """Calculate detailed diff statistics."""

    added_lines = sum(1 for line in diff_lines if line.startswith('+'))
    removed_lines = sum(1 for line in diff_lines if line.startswith('-'))
    changed_sections = sum(1 for line in diff_lines if line.startswith('@@'))

    # Calculate character and word differences
    text1 = '\n'.join(lines1)
    text2 = '\n'.join(lines2)

    char_diff = len(text2) - len(text1)
    word_diff = len(text2.split()) - len(text1.split())

    return {
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "changed_sections": changed_sections,
        "total_lines_old": len(lines1),
        "total_lines_new": len(lines2),
        "character_difference": char_diff,
        "word_difference": word_diff,
        "change_percentage": (added_lines + removed_lines) / max(len(lines1), len(lines2), 1) * 100
    }


def format_html_diff(diff_lines: List[str]) -> str:
    """Format diff lines as HTML with enhanced styling."""

    html_diff = []
    for line in diff_lines:
        escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        if line.startswith('+'):
            html_diff.append(f'<div class="diff-added"><span class="diff-marker">+</span>{escaped_line[1:]}</div>')
        elif line.startswith('-'):
            html_diff.append(f'<div class="diff-removed"><span class="diff-marker">-</span>{escaped_line[1:]}</div>')
        elif line.startswith('@@'):
            html_diff.append(f'<div class="diff-hunk">{escaped_line}</div>')
        else:
            html_diff.append(f'<div class="diff-context">{escaped_line}</div>')

    return '\n'.join(html_diff)


def format_inline_html(segments: List[Dict[str, Any]]) -> str:
    """Format inline diff segments as HTML."""

    html_parts = []
    for segment in segments:
        text = segment["text"].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        if segment["type"] == "equal":
            html_parts.append(text)
        elif segment["type"] == "delete":
            html_parts.append(f'<span class="diff-inline-removed">{text}</span>')
        elif segment["type"] == "insert":
            html_parts.append(f'<span class="diff-inline-added">{text}</span>')

    return ''.join(html_parts)


def extract_document_sections(text: str) -> List[Dict[str, Any]]:
    """Extract sections/clauses from document text."""
    
    # Simple section extraction based on headers
    # In a real implementation, this would be more sophisticated
    section_pattern = re.compile(r'(?:^|\n)((?:[A-Z][A-Z\s]+|[0-9]+\.\s+[A-Z][a-zA-Z\s]+):?)(?:\n|$)')
    
    sections = []
    last_pos = 0
    last_title = "PREAMBLE"
    
    # Find all section headers
    for match in section_pattern.finditer(text):
        # Add previous section
        if last_pos > 0:
            section_text = text[last_pos:match.start()].strip()
            if section_text:
                sections.append({
                    "title": last_title,
                    "text": section_text,
                    "start": last_pos,
                    "end": match.start()
                })
        
        last_title = match.group(1).strip()
        last_pos = match.end()
    
    # Add the last section
    if last_pos < len(text):
        section_text = text[last_pos:].strip()
        if section_text:
            sections.append({
                "title": last_title,
                "text": section_text,
                "start": last_pos,
                "end": len(text)
            })
    
    return sections


def compare_sections(old_sections: List[Dict[str, Any]], 
                    new_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare sections between document versions and identify changes."""
    
    changes = []
    
    # Create a map of section titles to sections for quick lookup
    old_section_map = {s["title"]: s for s in old_sections}
    new_section_map = {s["title"]: s for s in new_sections}
    
    # Find added sections
    for title, section in new_section_map.items():
        if title not in old_section_map:
            changes.append({
                "change_type": "added",
                "section_title": title,
                "old_text": None,
                "new_text": section["text"],
                "similarity_score": 0.0,
                "summary": f"New section '{title}' added"
            })
    
    # Find removed sections
    for title, section in old_section_map.items():
        if title not in new_section_map:
            changes.append({
                "change_type": "removed",
                "section_title": title,
                "old_text": section["text"],
                "new_text": None,
                "similarity_score": 0.0,
                "summary": f"Section '{title}' removed"
            })
    
    # Find modified sections
    for title, old_section in old_section_map.items():
        if title in new_section_map:
            new_section = new_section_map[title]
            
            # Skip if identical
            if old_section["text"] == new_section["text"]:
                continue
            
            # Calculate similarity
            similarity = calculate_text_similarity(old_section["text"], new_section["text"])
            
            # Generate summary of changes
            summary = summarize_changes(old_section["text"], new_section["text"])
            
            changes.append({
                "change_type": "modified",
                "section_title": title,
                "old_text": old_section["text"],
                "new_text": new_section["text"],
                "similarity_score": similarity,
                "summary": summary
            })
    
    return changes


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings."""
    # Simple similarity based on difflib
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def summarize_changes(old_text: str, new_text: str) -> str:
    """Generate a detailed summary of changes between two texts."""

    old_words = len(old_text.split())
    new_words = len(new_text.split())
    old_chars = len(old_text)
    new_chars = len(new_text)

    word_diff = new_words - old_words
    char_diff = new_chars - old_chars

    # Calculate similarity
    similarity = calculate_text_similarity(old_text, new_text)

    # Generate detailed summary
    summary_parts = []

    if word_diff > 0:
        summary_parts.append(f"expanded by {word_diff} words")
    elif word_diff < 0:
        summary_parts.append(f"reduced by {abs(word_diff)} words")

    if char_diff > 0:
        summary_parts.append(f"added {char_diff} characters")
    elif char_diff < 0:
        summary_parts.append(f"removed {abs(char_diff)} characters")

    if similarity < 0.5:
        summary_parts.append("major structural changes")
    elif similarity < 0.8:
        summary_parts.append("moderate changes")
    else:
        summary_parts.append("minor changes")

    if not summary_parts:
        return "Text modified with same word count"

    return "Text " + ", ".join(summary_parts) + f" (similarity: {similarity:.1%})"


def generate_semantic_diff(text1: str, text2: str) -> Dict[str, Any]:
    """Generate semantic-aware diff focusing on meaning rather than just text changes."""

    # Extract sections from both texts
    sections1 = extract_document_sections(text1)
    sections2 = extract_document_sections(text2)

    # Compare sections semantically
    section_changes = compare_sections(sections1, sections2)

    # Identify moved sections (sections that appear in different positions)
    moved_sections = identify_moved_sections(sections1, sections2)

    # Generate overall document diff
    document_diff = generate_document_diff(text1, text2)

    return {
        "document_diff": document_diff,
        "section_changes": section_changes,
        "moved_sections": moved_sections,
        "semantic_summary": generate_semantic_summary(section_changes, moved_sections),
        "change_impact": assess_change_impact(section_changes)
    }


def identify_moved_sections(old_sections: List[Dict[str, Any]],
                           new_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify sections that have been moved to different positions."""

    moved_sections = []

    # Create maps for quick lookup
    old_section_map = {s["title"]: s for s in old_sections}
    new_section_map = {s["title"]: s for s in new_sections}

    # Find sections that exist in both but at different positions
    for i, old_section in enumerate(old_sections):
        title = old_section["title"]
        if title in new_section_map:
            # Find position in new sections
            new_position = next((j for j, s in enumerate(new_sections) if s["title"] == title), -1)

            if new_position != -1 and new_position != i:
                # Section has moved
                moved_sections.append({
                    "section_title": title,
                    "old_position": i + 1,
                    "new_position": new_position + 1,
                    "text": old_section["text"],
                    "similarity_score": calculate_text_similarity(
                        old_section["text"],
                        new_section_map[title]["text"]
                    )
                })

    return moved_sections


def generate_semantic_summary(section_changes: List[Dict[str, Any]],
                             moved_sections: List[Dict[str, Any]]) -> str:
    """Generate a human-readable summary of semantic changes."""

    summary_parts = []

    # Count different types of changes
    added = sum(1 for c in section_changes if c["change_type"] == "added")
    removed = sum(1 for c in section_changes if c["change_type"] == "removed")
    modified = sum(1 for c in section_changes if c["change_type"] == "modified")
    moved = len(moved_sections)

    if added > 0:
        summary_parts.append(f"{added} section{'s' if added > 1 else ''} added")

    if removed > 0:
        summary_parts.append(f"{removed} section{'s' if removed > 1 else ''} removed")

    if modified > 0:
        summary_parts.append(f"{modified} section{'s' if modified > 1 else ''} modified")

    if moved > 0:
        summary_parts.append(f"{moved} section{'s' if moved > 1 else ''} moved")

    if not summary_parts:
        return "No significant structural changes detected"

    return ", ".join(summary_parts)


def assess_change_impact(section_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess the impact level of changes."""

    if not section_changes:
        return {"level": "none", "score": 0, "description": "No changes detected"}

    # Calculate impact score based on change types and similarity
    impact_score = 0

    for change in section_changes:
        if change["change_type"] == "added":
            impact_score += 3
        elif change["change_type"] == "removed":
            impact_score += 4
        elif change["change_type"] == "modified":
            # Impact based on similarity - lower similarity = higher impact
            similarity = change.get("similarity_score", 1.0)
            impact_score += (1 - similarity) * 5

    # Normalize impact score
    max_possible_score = len(section_changes) * 5
    normalized_score = min(impact_score / max_possible_score, 1.0) if max_possible_score > 0 else 0

    # Determine impact level
    if normalized_score < 0.2:
        level = "low"
        description = "Minor changes with minimal impact"
    elif normalized_score < 0.5:
        level = "medium"
        description = "Moderate changes requiring review"
    elif normalized_score < 0.8:
        level = "high"
        description = "Significant changes requiring careful review"
    else:
        level = "critical"
        description = "Major changes requiring immediate attention"

    return {
        "level": level,
        "score": normalized_score,
        "description": description,
        "total_changes": len(section_changes)
    }