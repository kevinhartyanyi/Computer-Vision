#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

/*
Lab#1 (10th Sept): Bouncing ball. Write a small game. 
The window itself is the field, it is bordered by three walls (left, right, top).
The fourth side is open, there is a wallet at that side .There is also a ball in the field, its speed is constant.
It bounces back from both the walls and the wallets and racket. The racket can be controlled by mouse and/or keyboard.
If the ball falls down, the game exists.


Score: 5%
If both mouse and keyboard can be used for controlling: +1%

Deadline: end of next laboratory (Thursday)
*/

using namespace cv;
using namespace std;

#define WIDTH 800
#define HEIGHT 600
#define THICKNESS 20
#define BASEWIDTH THICKNESS*4
#define BLACK Scalar(0, 0, 0)
#define SPEED 10


Mat image;

int L1Norm(int x, int y) {
	return abs(x) + abs(y);
}

int L1Norm(Vec2i vect) {
	return abs(vect[0]) + abs(vect[1]);
}

class Drawable
{
public:
	Drawable(Point start, Point end) : _start(start), _end(end) {};
	virtual void draw() = 0;
protected:
	Point _start;
	Point _end;
};

class Wall : public Drawable
{
public:
	Wall(Point start, Point end) : Drawable(start, end) {};
	Point getStart() const { return _start; }
	Point getEnd() const { return _end; }
	void draw() override {
		rectangle(image, _start, _end, BLACK, CV_FILLED);
	}

private:
};



class Player : public Drawable
{
public:
	Player(Point start, Point end) : Drawable(start, end) {};
	Point getStart() const { return _start; }
	Point getEnd() const { return _end; }
	void draw() override {
		rectangle(image, _start, _end, BLACK, CV_FILLED);
	}

	void changePosition(Point start, Point end) {
		_start = start;
		_end = end;
	}

	void moveRight() {
		if (_end.x < WIDTH) 
		{
			_start.x += SPEED;
			_end.x += SPEED;
		}
	}
	void moveLeft() {
		if (_start.x > 0)
		{
			_start.x -= SPEED;
			_end.x -= SPEED;
		}
	}	

private:
	
};

class Ball : public Drawable
{
public:
	Ball(Point start, Point end, Player* player) : Drawable(start, end), _player(player) {
		_direction = Vec2i(0, SPEED);
	};
	Point getStart() const { return _start; }
	Point getEnd() const { return _end; }
	void draw() override {
		checkCollision();
		move(_direction[0], _direction[1]);
		rectangle(image, _start, _end, BLACK, CV_FILLED);
	}

	bool gameOver = false;



private:

	Vec2i _direction;
	Player* _player;

	void checkCollision() {
		if (_start.x <= THICKNESS) // Left Wall
		{
			_direction[0] = -_direction[0];
		}
		else if (_end.x >= WIDTH - THICKNESS) // Right Wall
		{
			_direction[0] = -_direction[0];
		}
		else if (_start.y <= THICKNESS) // Up Wall
		{
			_direction[1] = -_direction[1];
		}
		else if (overlapPlayer()) {
			int playerWidth = _player->getEnd().x - _player->getStart().x;
			int ballWidth = _end.x - _start.x;
			int playerCenter = _player->getStart().x + playerWidth / 2;
			int ballCenter = _start.x + ballWidth / 2;
			std::cout << "playerCenter: " << playerCenter << std::endl;
			std::cout << "ballCenter: " << ballCenter << std::endl;
			Vec2i baseVector(ballCenter - playerCenter, (playerWidth / 2) / (abs(playerCenter - ballCenter) + 0.00001));


			std::cout << "baseVector: " << baseVector << std::endl;
			std::cout << "magnitude: " << L1Norm(baseVector) << std::endl;			

			int x = round((baseVector[0] / (float)L1Norm(baseVector)) * (SPEED - 1));
			int y = round((baseVector[1] / (float)L1Norm(baseVector)) * (SPEED - 1)) + 1;

			std::cout << "x: " << x << std::endl;
			std::cout << "y: " << y << std::endl;

			_direction = Vec2i(x,-y);
		}
		else if (_end.y >= HEIGHT - THICKNESS) // Game Over
		{
			gameOver = true;
		}	

	}

	bool overlapPlayer()
	{
		bool re = false;
		if (_start.x >= _player->getStart().x && _start.x <= _player->getEnd().x && _end.y >= _player->getStart().y
			|| _end.x >= _player->getStart().x && _end.x <= _player->getEnd().x && _end.y >= _player->getStart().y)
		{
			re = true;
		}
		return re;
	}

	void move(int x, int y) {
		_start.x += x;
		_start.y += y;
		_end.x += x;
		_end.y += y;
	}

	void changePosition(Point start, Point end) {
		_start = start;
		_end = end;
	}
	void moveRight(int length) {
		_start.x += length;
		_end.x += length;
	}
	void moveLeft(int length) {
		_start.x -= length;
		_end.x -= length;
	}
	void moveDown(int length) {
		_start.y += length;
		_end.y += length;
	}
	void moveUp(int length) {
		_start.y -= length;
		_end.y -= length;
	}
};




class Drawer
{
public:
	template<typename First, typename ... Rest>
	void addDrawable(First& first, Rest&... rest) {
		elements.push_back(first);
		addDrawable(rest...);
	}
	void addDrawable() {}


	void draw() {
		for (Drawable* elem : elements) {
			elem->draw();
		}
	}


private:
	vector<Drawable*> elements;
};


Drawer drawer;


void redraw() {
	rectangle(image, Point(0, 0), Point(WIDTH, HEIGHT), Scalar(255, 255, 255), CV_FILLED);
	drawer.draw();
	imshow("Display window", image);                   // Show our image inside it.
}

void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		redraw();
	}

}


int main(int argc, char** argv)
{


	image = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	Drawable* left = new Wall(Point(0, 0), Point(THICKNESS, HEIGHT));
	Drawable* up = new Wall(Point(0, 0), Point(WIDTH, THICKNESS));
	Drawable* right = new Wall(Point(WIDTH - THICKNESS, 0), Point(WIDTH, HEIGHT));

	Drawable* player = new Player(Point((WIDTH/2) - BASEWIDTH, HEIGHT-THICKNESS), Point((WIDTH / 2) + BASEWIDTH, HEIGHT));

	Drawable* ball = new Ball(Point((WIDTH / 2) - THICKNESS, (HEIGHT / 2) - THICKNESS), Point((WIDTH / 2) + THICKNESS, (HEIGHT / 2) + THICKNESS), (Player*)(player));

	
	drawer.addDrawable(left, up, right, ball, player);


	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	setMouseCallback("Display window", MouseCallBackFunc, NULL);



	imshow("Display window", image);                   // Show our image inside it.


	Player* base = (Player*)player;
	Ball* game = (Ball*)ball;

	int key;
	while (true) {
		key = cvWaitKey(5);
		//iksz++;
		if (game->gameOver) break;
		
		switch (key) {
		case 'a':
			base->moveLeft();
			break;
		case 'd':
			base->moveRight();
			break;
		}
		redraw();
	}

	return 0;
}
