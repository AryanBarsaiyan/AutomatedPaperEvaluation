import { Component, OnInit } from '@angular/core';
import { sample, sampleTime } from 'rxjs';
import axios from 'axios';
import { SpinnerService } from './spinner/spinner.service';
import {
  FormArray,
  FormBuilder,
  FormControl,
  FormGroup,
  Validators,
} from '@angular/forms';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent implements OnInit {
  orderForm: any = FormGroup;
  rows: any = FormArray;

  answerForm: FormGroup;
  sampleAnswer = '';
  sampleQuestion = '';
  constructor(
    private formBuilder: FormBuilder,
    private spinnerService: SpinnerService
  ) {
    this.answerForm = this.formBuilder.group({
      sampleQuestion: '',
      sampleAnswer: '',
      answer: this.formBuilder.array([]),
    });
  }

  ngOnInit() {}

  result: number = 0;
  resultStatus: boolean = false;

  // //https://medium.com/tech-insights/how-to-deploy-angular-apps-to-github-pages-gh-pages-896c4e10f9b4
  // // for deployment!
  payload: object = { sample: '', test: '' };

  CalculateScore(sampleQuestion: string,sampleAnswer: string, testAnswer: string, index: number) {
    var matchScore: number = 0;
    if (sampleAnswer == '' || sampleQuestion == '') alert('Please enter Editorial Answer  or Question !');
    else {
      this.payload = { Question: sampleQuestion,editorialAnswer: sampleAnswer, userAnswer: testAnswer };

      // console.log(this.payload);

      this.spinnerService.requestStarted();

      axios.defaults.headers.post['Content-Type'] =
        'application/json;charset=utf-8';
      axios.defaults.headers.post['Access-Control-Allow-Origin'] = '*';
      axios
        .post('http://127.0.0.1:5000/', this.payload)
        .then((response) => {
          // console.log(response['data'] +"%")
          this.spinnerService.requestEnded();
          matchScore = response['data'] * 100;

          this.answer()
            .at(index)
            .setValue({ testAnswer: testAnswer, score: matchScore.toFixed(2) });
        })
        .catch((error) => {
          this.spinnerService.resetSpinner();
          console.error('There was an error!', error);
          matchScore = -1;
          this.answer()
            .at(index)
            .setValue({ testAnswer: testAnswer, score: matchScore });
          alert(
            "didn't able calculate one of the test answer and it's Match score will be -1"
          );
        });
      // console.log( this.payload );
    }
  }

  answer(): FormArray {
    return this.answerForm.get('answer') as FormArray;
  }

  newAnswer(): FormGroup {
    return this.formBuilder.group({
      testAnswer: '',
      score: '',
    });
  }

  addAnswer() {
    this.answer().push(this.newAnswer());
  }

  removeAnswer(i: number) {
    this.answer().removeAt(i);
  }

  onSubmit() {
    // this.spinnerService.requestStarted();
    // this.answer().at(0).value['score'] = "10"; // this need to be done, how to set score value?  , setValue
    // this.answer().at(0).setValue({'testAnswer':"hmm", 'score': this.result }) // this how to set value
    // console.log( this.answer().at(0).value  );
    const sampleQuestion = this.answerForm.value.sampleQuestion;

    const sampleAnswer = this.answerForm.value.sampleAnswer;
    let answerCount = this.answerForm.value.answer.length;
    // console.log( answerCount );
    // console.log( sampleAnswer);
    // console.log(this.answerForm.value);

    for (let i = 0; i < answerCount; i++) {
      const testAnswer = this.answer().at(i).value['testAnswer'];
      const matchScore = this.CalculateScore(sampleQuestion ,sampleAnswer, testAnswer, i);
      // console.log( testAnswer);
      // console.log( matchScore );
      // this.answer().at(i).setValue({'testAnswer': testAnswer, 'score': matchScore });
    }
  }
}