"""
Enhanced Document Retriever for Complex English Queries

This module provides advanced search capabilities that can handle complex questions
without relying on hardcoded phrases, specifically optimized for English queries.
"""

import re
import logging
from typing import List, Dict, Any, Optional
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.connection import async_session

logger = logging.getLogger(__name__)


class EnhancedDocumentRetriever:
    """Advanced document retriever optimized for complex English queries."""
    
    def __init__(self):
        self.similarity_threshold = 0.3
        
        # Comprehensive synonym dictionary for legal and technical terms
        self.concept_synonyms = {
            'accept': ['agree', 'consent', 'approve', 'acknowledge', 'embrace', 'adopt'],
            'acceptance': ['agreement', 'consent', 'approval', 'acknowledgment', 'adoption'],
            'responsibility': ['obligation', 'duty', 'liability', 'accountability', 'commitment'],
            'responsibilities': ['obligations', 'duties', 'commitments', 'requirements', 'mandates'],
            'obligations': ['duties', 'responsibilities', 'commitments', 'requirements', 'mandates'],
            'rights': ['privileges', 'entitlements', 'permissions', 'authorities', 'powers'],
            'license': ['permit', 'authorization', 'permission', 'grant', 'allow'],
            'licensed': ['permitted', 'authorized', 'allowed', 'granted', 'entitled'],
            'distribute': ['provide', 'share', 'disseminate', 'deliver', 'supply'],
            'distribution': ['sharing', 'dissemination', 'delivery', 'provision', 'supply'],
            'redistribute': ['reshare', 'redisseminate', 'redelivery', 'resupply'],
            'modification': ['change', 'alteration', 'amendment', 'update', 'revision'],
            'modifications': ['changes', 'alterations', 'amendments', 'updates', 'revisions'],
            'sublicense': ['sublicensing', 'sub-license', 'relicense', 'secondary license'],
            'requirements': ['conditions', 'obligations', 'mandates', 'specifications', 'criteria'],
            'comply': ['adhere', 'conform', 'observe', 'follow', 'satisfy'],
            'compliance': ['adherence', 'conformity', 'observance', 'satisfaction'],
            'noncompliance': ['violation', 'breach', 'non-adherence', 'non-conformity'],
            'cure': ['fix', 'remedy', 'correct', 'resolve', 'address'],
            'terminate': ['end', 'cancel', 'discontinue', 'cease', 'stop'],
            'termination': ['ending', 'cancellation', 'discontinuation', 'cessation'],
            'governing': ['controlling', 'ruling', 'applicable', 'relevant'],
            'warranty': ['guarantee', 'assurance', 'promise', 'commitment'],
            'indemnity': ['protection', 'compensation', 'reimbursement', 'coverage'],
            'export': ['international', 'foreign', 'overseas', 'external'],
            'endorsement': ['approval', 'support', 'backing', 'recommendation'],
            'commercial': ['business', 'trade', 'market', 'profit'],
            'advantage': ['benefit', 'edge', 'superiority', 'gain'],
            'registration': ['enrollment', 'signup', 'recording', 'listing'],
            'representations': ['statements', 'declarations', 'assertions', 'claims'],
            'purpose': ['objective', 'goal', 'aim', 'intention', 'reason'],
            'subject': ['covered', 'relevant', 'applicable', 'related'],
            'contributor': ['provider', 'supplier', 'giver', 'author'],
            'recipient': ['receiver', 'user', 'beneficiary', 'party'],
            'agency': ['organization', 'department', 'bureau', 'authority'],
            'government': ['federal', 'state', 'public', 'official'],
            'trigger': ['cause', 'initiate', 'activate', 'start', 'prompt'],
            'perform': ['execute', 'carry out', 'conduct', 'do', 'implement'],
            'activities': ['actions', 'operations', 'tasks', 'work', 'processes'],
            'define': ['specify', 'describe', 'explain', 'clarify', 'identify'],
            'provide': ['supply', 'give', 'offer', 'furnish', 'deliver'],
            'include': ['contain', 'comprise', 'incorporate', 'encompass'],
            'remove': ['delete', 'eliminate', 'take away', 'extract'],
            'request': ['ask for', 'seek', 'require', 'demand'],
            'offer': ['provide', 'give', 'supply', 'present'],
            'fail': ['not succeed', 'be unable', 'neglect', 'omit'],
            'software': ['code', 'program', 'application', 'system'],
            'source code': ['source', 'code', 'programming code'],
            'copyright': ['intellectual property', 'authorship', 'ownership'],
            'notice': ['notification', 'announcement', 'statement', 'message'],
            'patent': ['intellectual property', 'invention', 'IP'],
            'when': ['if', 'upon', 'during', 'while', 'as'],
            'conditions': ['circumstances', 'situations', 'requirements', 'terms'],
            'happens': ['occurs', 'takes place', 'arises', 'comes about'],
            'stated': ['mentioned', 'specified', 'declared', 'indicated'],
            'limitations': ['restrictions', 'constraints', 'bounds', 'limits'],
            'placed': ['imposed', 'applied', 'set', 'established'],
        }
        
        # Stop words for filtering
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }

    async def retrieve_relevant_chunks(
        self, 
        query: str, 
        limit: int = 5,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval with complex query understanding."""
        
        try:
            logger.info(f"Enhanced search for query: '{query}'")
            
            # Analyze the query
            query_analysis = self._analyze_complex_query(query)
            
            # Build enhanced search patterns
            search_patterns = self._build_search_patterns(query_analysis)
            
            # Execute search with multiple strategies
            results = await self._execute_enhanced_search(
                search_patterns, limit, document_ids, chunk_types
            )
            
            logger.info(f"Enhanced search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {e}")
            return []

    def _analyze_complex_query(self, query: str) -> Dict[str, Any]:
        """Analyze complex queries to extract intent, entities, and concepts."""
        query_lower = query.lower().strip()
        
        # Detect query intent
        intent = self._detect_query_intent(query_lower)
        
        # Extract meaningful terms
        key_terms = self._extract_meaningful_terms(query_lower)
        
        # Extract important phrases
        phrases = self._extract_important_phrases(query_lower)
        
        # Generate search variations
        search_variations = self._generate_search_variations(key_terms)
        
        return {
            'original_query': query,
            'intent': intent,
            'key_terms': key_terms,
            'phrases': phrases,
            'search_variations': search_variations
        }
    
    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect the intent and type of the query."""
        intent = {
            'type': 'general',
            'focus': [],
            'question_words': [],
            'numbered_items': [],
            'technical_frameworks': []
        }

        # Question type detection
        if any(phrase in query for phrase in ['what are', 'what is', 'define']):
            intent['type'] = 'definition'
        elif any(phrase in query for phrase in ['when', 'under what conditions', 'if']):
            intent['type'] = 'conditions'
        elif any(phrase in query for phrase in ['how', 'what activities', 'what actions']):
            intent['type'] = 'process'
        elif any(phrase in query for phrase in ['can', 'may', 'able to']):
            intent['type'] = 'permission'
        elif any(phrase in query for phrase in ['what happens', 'what occurs']):
            intent['type'] = 'consequence'
        elif any(phrase in query for phrase in ['requirements', 'must', 'shall']):
            intent['type'] = 'obligation'

        # Detect numbered items (stages, steps, phases)
        import re
        numbered_patterns = [
            r'(\d+)(?:st|nd|rd|th)?\s+(stage|step|phase|level|tier)',
            r'(stage|step|phase|level|tier)\s+(\d+)',
            r'(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(stage|step|phase)',
            r'(stage|step|phase)\s+(one|two|three|four|five|six|seven|eight|nine|ten)'
        ]

        for pattern in numbered_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    intent['numbered_items'].extend([item.lower() for item in match if item])
                else:
                    intent['numbered_items'].append(match.lower())

        # Detect technical frameworks
        technical_frameworks = [
            'cyber kill chain', 'kill chain', 'mitre att&ck', 'nist framework',
            'iso 27001', 'owasp', 'sans', 'cis controls', 'pci dss'
        ]

        for framework in technical_frameworks:
            if framework in query:
                intent['technical_frameworks'].append(framework)

        # Extract focus areas (expanded for technical content)
        all_focuses = [
            # Legal terms
            'patent', 'copyright', 'license', 'agreement', 'contract',
            'modification', 'distribution', 'sublicense', 'warranty',
            'indemnity', 'compliance', 'termination', 'export', 'commercial',
            'government', 'agency', 'contributor', 'recipient', 'software',
            # Technical/Security terms
            'reconnaissance', 'weaponization', 'delivery', 'exploitation',
            'installation', 'command', 'control', 'actions', 'objectives',
            'malware', 'payload', 'vulnerability', 'attack', 'intrusion',
            'cybersecurity', 'security', 'threat', 'adversary', 'target'
        ]

        for focus in all_focuses:
            if focus in query:
                intent['focus'].append(focus)

        return intent
    
    def _extract_meaningful_terms(self, query: str) -> List[str]:
        """Extract meaningful terms, filtering out stop words."""
        # Remove punctuation and split
        cleaned_query = re.sub(r'[^\w\s]', ' ', query)
        words = cleaned_query.split()
        
        # Filter out stop words and short words
        meaningful_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return meaningful_words
    
    def _extract_important_phrases(self, query: str) -> List[str]:
        """Extract important compound phrases."""
        phrases = []

        # Important compound terms (legal and technical)
        compound_patterns = [
            # Legal terms
            'subject software', 'source code', 'government agency',
            'patent rights', 'copyright notice', 'open source',
            'user registration', 'commercial advantage', 'export license',
            'governing law', 'intellectual property', 'non-commercial use',
            # Technical/Security terms
            'cyber kill chain', 'kill chain', 'command and control',
            'command & control', 'actions on objectives', 'lateral movement',
            'privilege escalation', 'data exfiltration', 'persistence mechanism',
            'initial access', 'execution phase', 'defense evasion',
            'credential access', 'discovery phase', 'collection phase',
            'impact phase', 'threat actor', 'attack vector', 'attack surface'
        ]

        for pattern in compound_patterns:
            if pattern in query.lower():
                phrases.append(pattern)

        # Extract numbered stage/step phrases
        import re
        numbered_phrase_patterns = [
            r'(\d+)(?:st|nd|rd|th)?\s+stage\s+(?:of\s+)?(?:the\s+)?(\w+(?:\s+\w+)*)',
            r'stage\s+(\d+)\s+(?:of\s+)?(?:the\s+)?(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+stage\s+(\d+)',
            r'(\w+(?:\s+\w+)*)\s+(\d+)(?:st|nd|rd|th)?\s+stage'
        ]

        for pattern in numbered_phrase_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                if isinstance(match, tuple):
                    # Reconstruct meaningful phrases from matches
                    phrase_parts = [part.strip() for part in match if part.strip()]
                    if len(phrase_parts) >= 2:
                        phrases.append(' '.join(phrase_parts))

        return phrases
    
    def _generate_search_variations(self, key_terms: List[str]) -> Dict[str, List[str]]:
        """Generate search variations using synonyms and related terms."""
        variations = {
            'exact': key_terms,
            'synonyms': [],
            'related': [],
            'numbered_alternatives': []
        }

        for term in key_terms:
            term_lower = term.lower()

            # Add synonyms from predefined dictionary
            if term_lower in self.concept_synonyms:
                variations['synonyms'].extend(self.concept_synonyms[term_lower])

            # Generate numbered alternatives for ordinal numbers
            numbered_alternatives = self._generate_numbered_alternatives(term_lower)
            variations['numbered_alternatives'].extend(numbered_alternatives)

        return variations

    def _generate_numbered_alternatives(self, term: str) -> List[str]:
        """Generate alternative representations for numbered items."""
        alternatives = []

        # Ordinal to cardinal mappings
        ordinal_to_cardinal = {
            'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
            'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10',
            '1st': '1', '2nd': '2', '3rd': '3', '4th': '4', '5th': '5',
            '6th': '6', '7th': '7', '8th': '8', '9th': '9', '10th': '10'
        }

        cardinal_to_ordinal = {v: k for k, v in ordinal_to_cardinal.items()}

        # Check if term contains ordinal numbers
        for ordinal, cardinal in ordinal_to_cardinal.items():
            if ordinal in term:
                # Generate alternative with cardinal number
                alternatives.append(term.replace(ordinal, cardinal))
                # Generate alternative with different ordinal format
                if ordinal.endswith(('st', 'nd', 'rd', 'th')):
                    word_form = {
                        '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth', '5th': 'fifth',
                        '6th': 'sixth', '7th': 'seventh', '8th': 'eighth', '9th': 'ninth', '10th': 'tenth'
                    }.get(ordinal)
                    if word_form:
                        alternatives.append(term.replace(ordinal, word_form))

        # Check if term contains cardinal numbers
        for cardinal, ordinal in cardinal_to_ordinal.items():
            if cardinal in term and not any(char.isalpha() for char in term.replace(cardinal, '')):
                # Generate alternatives with ordinal forms
                alternatives.append(term.replace(cardinal, ordinal))
                alternatives.append(term.replace(cardinal, cardinal + 'st' if cardinal == '1' else
                                                cardinal + 'nd' if cardinal == '2' else
                                                cardinal + 'rd' if cardinal == '3' else
                                                cardinal + 'th'))

        return alternatives
    
    def _build_search_patterns(self, query_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build search patterns based on query analysis."""
        patterns = {
            'exact_phrases': query_analysis['phrases'],
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }

        # High priority: key terms related to intent focus and technical frameworks
        intent_focus = query_analysis['intent']['focus']
        technical_frameworks = query_analysis['intent'].get('technical_frameworks', [])
        numbered_items = query_analysis['intent'].get('numbered_items', [])

        for term in query_analysis['key_terms']:
            if (term.lower() in intent_focus or
                any(framework in term.lower() for framework in technical_frameworks) or
                any(item in term.lower() for item in numbered_items)):
                patterns['high_priority'].append(term)

        # Add numbered alternatives to high priority for numbered queries
        if numbered_items:
            patterns['high_priority'].extend(query_analysis['search_variations']['numbered_alternatives'])

        # Medium priority: remaining key terms and technical framework terms
        for term in query_analysis['key_terms']:
            if term not in patterns['high_priority']:
                patterns['medium_priority'].append(term)

        # Add technical framework terms to medium priority
        for framework in technical_frameworks:
            if framework not in patterns['high_priority']:
                patterns['medium_priority'].append(framework)

        # Intent-based keywords
        intent_type = query_analysis['intent']['type']
        intent_keywords = self._get_intent_keywords(intent_type)
        patterns['medium_priority'].extend(intent_keywords)

        # Low priority: synonyms and variations
        patterns['low_priority'].extend(query_analysis['search_variations']['synonyms'][:10])

        return patterns
    
    def _get_intent_keywords(self, intent_type: str) -> List[str]:
        """Get keywords associated with specific intent types."""
        intent_keywords = {
            'definition': ['means', 'refers to', 'defined as', 'definition'],
            'conditions': ['if', 'when', 'upon', 'provided that', 'subject to'],
            'permission': ['may', 'can', 'permitted', 'authorized', 'licensed'],
            'obligation': ['shall', 'must', 'required', 'obligated', 'duty'],
            'consequence': ['result', 'lead to', 'cause', 'trigger', 'happens']
        }
        return intent_keywords.get(intent_type, [])
    
    async def _execute_enhanced_search(
        self, 
        search_patterns: Dict[str, List[str]], 
        limit: int,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Execute enhanced search using the generated patterns."""
        
        async with async_session() as session:
            where_conditions = []
            params = {"limit": limit}
            
            if document_ids:
                where_conditions.append("d.id = ANY(:document_ids)")
                params["document_ids"] = document_ids
            
            if chunk_types:
                where_conditions.append("e.chunk_type = ANY(:chunk_types)")
                params["chunk_types"] = chunk_types
            
            # Build search conditions
            search_conditions = []
            param_count = 0
            
            # Exact phrases (highest priority)
            for phrase in search_patterns['exact_phrases']:
                search_conditions.append(f"e.chunk_text ILIKE :pattern_{param_count}")
                params[f"pattern_{param_count}"] = f"%{phrase}%"
                param_count += 1
            
            # High priority terms
            for term in search_patterns['high_priority']:
                search_conditions.append(f"e.chunk_text ILIKE :pattern_{param_count}")
                params[f"pattern_{param_count}"] = f"%{term}%"
                param_count += 1
            
            # Medium priority terms
            for term in search_patterns['medium_priority'][:15]:  # Limit to avoid too many conditions
                search_conditions.append(f"e.chunk_text ILIKE :pattern_{param_count}")
                params[f"pattern_{param_count}"] = f"%{term}%"
                param_count += 1
            
            # Low priority terms
            for term in search_patterns['low_priority'][:10]:  # Limit further
                search_conditions.append(f"e.chunk_text ILIKE :pattern_{param_count}")
                params[f"pattern_{param_count}"] = f"%{term}%"
                param_count += 1
            
            if not search_conditions:
                return []
            
            base_where = " AND ".join(where_conditions) if where_conditions else "1=1"
            search_where = " OR ".join(search_conditions)
            
            # Calculate scoring weights
            exact_phrase_count = len(search_patterns['exact_phrases'])
            high_priority_count = len(search_patterns['high_priority'])
            medium_priority_count = len(search_patterns['medium_priority'][:15])
            low_priority_count = len(search_patterns['low_priority'][:10])
            
            # Build scoring logic
            scoring_logic = []
            
            if exact_phrase_count > 0:
                exact_conditions = []
                for i in range(exact_phrase_count):
                    exact_conditions.append(f"e.chunk_text ILIKE :pattern_{i}")
                scoring_logic.append(f"CASE WHEN ({' OR '.join(exact_conditions)}) THEN 100 ELSE 0 END")
            
            if high_priority_count > 0:
                high_start = exact_phrase_count
                high_end = high_start + high_priority_count
                high_conditions = []
                for i in range(high_start, high_end):
                    high_conditions.append(f"e.chunk_text ILIKE :pattern_{i}")
                scoring_logic.append(f"CASE WHEN ({' OR '.join(high_conditions)}) THEN 80 ELSE 0 END")
            
            if medium_priority_count > 0:
                medium_start = exact_phrase_count + high_priority_count
                medium_end = medium_start + medium_priority_count
                medium_conditions = []
                for i in range(medium_start, medium_end):
                    medium_conditions.append(f"e.chunk_text ILIKE :pattern_{i}")
                scoring_logic.append(f"CASE WHEN ({' OR '.join(medium_conditions)}) THEN 60 ELSE 0 END")
            
            if low_priority_count > 0:
                low_start = exact_phrase_count + high_priority_count + medium_priority_count
                low_end = low_start + low_priority_count
                low_conditions = []
                for i in range(low_start, low_end):
                    low_conditions.append(f"e.chunk_text ILIKE :pattern_{i}")
                scoring_logic.append(f"CASE WHEN ({' OR '.join(low_conditions)}) THEN 40 ELSE 0 END")
            
            final_scoring = " + ".join(scoring_logic) if scoring_logic else "0"
            
            query_sql = f"""
                SELECT 
                    e.id as embedding_id,
                    e.chunk_id,
                    e.chunk_text,
                    e.chunk_type,
                    e.start_char,
                    e.end_char,
                    d.id as document_id,
                    d.filename,
                    d.file_type,
                    d.upload_date,
                    ({final_scoring}) as similarity_score
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE {base_where} AND ({search_where})
                ORDER BY similarity_score DESC, e.created_at DESC
                LIMIT :limit
            """
            
            result = await session.execute(text(query_sql), params)
            rows = result.fetchall()
            
            return self._format_results(rows)
    
    def _format_results(self, rows) -> List[Dict[str, Any]]:
        """Format database results into standard format."""
        results = []
        for row in rows:
            chunk_data = {
                "id": str(row.embedding_id),
                "chunk_id": row.chunk_id,
                "text": row.chunk_text,
                "chunk_type": row.chunk_type,
                "start_char": row.start_char,
                "end_char": row.end_char
            }
            
            document_data = {
                "id": str(row.document_id),
                "filename": row.filename,
                "file_type": row.file_type,
                "upload_date": row.upload_date.isoformat() if row.upload_date else None
            }
            
            results.append({
                "chunk": chunk_data,
                "document": document_data,
                "similarity_score": float(row.similarity_score) / 100.0  # Normalize to 0-1
            })
        
        return results

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        async with async_session() as session:
            query_sql = """
                SELECT 
                    e.id as embedding_id,
                    e.chunk_id,
                    e.chunk_text,
                    e.chunk_type,
                    e.start_char,
                    e.end_char,
                    d.id as document_id,
                    d.filename,
                    d.file_type,
                    d.upload_date
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE d.id = :document_id
                ORDER BY e.start_char ASC
            """
            
            result = await session.execute(text(query_sql), {"document_id": document_id})
            rows = result.fetchall()
            
            return self._format_results(rows) 