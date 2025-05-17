# from phisher_man.tackler.tackler import tackle
from phisher_man.tackler import tackle
from phisher_man.bobber import bob, bob_all
# from phisher_man.big_bait import big_bait


def main() -> None:
    print("Hello from phisher-man!")


def tackler():
    tackle()


def bobber():
    bob()


def bobber_all():
    bob_all()


def big_baiter():
    bob("./combined_text_phishing_model.pkl")
