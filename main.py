from Summarizers import SummarizeTLDR
from Summarizers import SummarizeTextRank
from Summarizers import SummarizeLSA
from Summarizers import SummarizeBart
from Summarizers import SummarizeEnsemble
# from QA_Generation import QuestgenMachineLearning

def main():
    file = open('Data/book.txt','r')
    contents= file.read()
    # SummarizeTLDR.summarizer_tldr()
    # SummarizeTextRank.generate_summary(contents)
    # SummarizeLSA.generate_summary(contents)
    SummarizeBart.generate_summary(contents)
    # SummarizeEnsemble.generate_summary(contents)
    # QuestgenMachineLearning.generate_qa_pairs()
main()

