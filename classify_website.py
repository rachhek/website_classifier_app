from clint.arguments import Args
from clint.textui import puts, colored, indent
from pyfiglet import Figlet
import lib


def main():
    f = Figlet(font='slant')
    print(f.renderText('Website Classifier'))
    args = Args()
    with indent(4, quote='>>>'):
        puts(colored.blue("This program classifies program into 11 different categories. The model is trained on 59k+ websites using tfidf features and Logistic regression model."))
        puts(colored.blue('Args received: ') + str(args.all))
        print()

    website_url = args[0]

    # Initializing the WebsiteClassifierTool Object to predic the website class.
    wct = lib.WebsiteClassifierTool()
    result = wct.predict(website_url)

    # print the final results of the classification
    print(colored.yellow("Result:"))
    print(colored.green(result))


if __name__ == "__main__":
    main()
