import pandas as pd

def convert_labels(csv_file_path, output_csv_path):
    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)

    # "safe"를 1로, "unsafe"를 0으로 바꾸기
    df["Label"] = df["Label"].apply(lambda x: 1 if x == "safe" else 0 if x == "unsafe" else x)

    # 바뀐 데이터프레임을 새로운 CSV 파일로 저장
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    # 원본 CSV 파일 경로와 바꾼 후 저장할 CSV 파일 경로 지정
    input_csv_path = "C://fof1//test/safe_unsafe_list.csv"
    output_csv_path = "C://fof1//test/safe_unsafe_list_out.csv"

    # 함수 호출
    convert_labels(input_csv_path, output_csv_path)