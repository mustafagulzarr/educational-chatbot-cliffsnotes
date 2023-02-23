import SummarizeTLDR
import SummarizeTextRank
import SummarizeLSA

def main():
    file = open('book.txt','r')
    contents= file.read()
    # SummarizeTLDR.summarizer_tldr()
    # SummarizeTextRank.generate_summary(contents)
    SummarizeLSA.generate_summary(contents)

main()

