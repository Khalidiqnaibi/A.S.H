class PromptTemplate:
    def __init__(self, role, question, context, language, constraints, output, rules):
        self._role = role
        self._language = language
        self._rules = rules
        self._constraints = constraints
        self._output = output
        self._context = context
        self._question = question

        donts_formated = [f"- {item}" for item in self._constraints]
        donts_formated = '\n '.join(donts_formated)

        do_formated = [f"- {item}" for item in self._rules]
        do_formated = '\n '.join(do_formated)

        self.command = f"""You are a {self._role}. Use ONLY the provided CONTEXT to answer the QUESTION. 
        Be clear and thorough.

        IMPORTANT:
        - Answer in plain {self._language}, full sentences.
        {do_formated}
        {donts_formated}
        - Do NOT output only YES or NO or any single-token classification.
        - Do NOT output JSON, YAML, or any machine-only format.
        - Give the output in this format: {self._output}.

        CONTEXT:
        {self._context}

        QUESTION: {self._question}

        Answer:"""

    def get_command(self) -> str:
        return self.command

                

