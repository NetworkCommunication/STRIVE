def main():
    with open(' WGAN_result', 'r') as f:
        file = f.readlines()[0].strip()
        print(file)
    num = 0
    print('[', end='')
    for item in file.split(','):
        if num % 2 == 0:
            print(item[-6:], end=', ')
        num += 1
    print(']')

if __name__ == '__main__':
    main()